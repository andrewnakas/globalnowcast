"""Train the global 6-48h GFS REFC correction against observed rain rate.

Pairs come from ml/data/build_gfs_pairs.py: the *served* GFS REFC forecast at
leads 6-48h, with the RRQPE satellite observation at the valid time as target -
training on exactly what the hourly job corrects, which every failed gate this
project has run says is the only representation that transfers.

Input channels: GFS dBZ, lead/48, |lat|/90. Lead because the bias grows with
forecast age; latitude because GFS is measurably worst in the tropics (CSI 0.163
vs ~0.2 midlat). Crops are NOT wet-selected - the hourly model's tile-vs-frame
failure showed dry sky must be in the training distribution.

Two structural protections of the measured "never frequency-match GFS" rule:
the correction is spatial/conditional (a residual field, not a marginal
recalibration), and pipeline/main.py applies it only at leads beyond the blend's
reach, so the 0-4.5h product keeps raw GFS by construction.

    python ml/train_correction.py --pairs ml/gfs_pairs --epochs 40

Gate (before any ONNX ships): pooled held-out CSI at 5/10/20/30 dBZ must improve
over raw GFS, wet-area ratio must move toward 1 without undershooting, and the
blend-with-corrected must be >= blend-with-raw on archived cases.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "verify"))

from metrics import contingency, csi  # noqa: E402
from model import POOL_DIVISOR, RefcUNet  # noqa: E402

U8_LO, U8_HI = -30.0, 60.0
THRESHOLDS = (5.0, 10.0, 20.0, 30.0)
LAT = np.linspace(90.0, -90.0, 721, dtype=np.float32)


def from_u8(a):
    return a.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO


def load_pairs(paths):
    """[(gfs dBZ, lead_h, obs dBZ, mask)] per lead field, float32/bool."""
    out = []
    for p in paths:
        with np.load(p) as z:
            obs = from_u8(z["obs"])
            mask = z["mask"]
            for k in z.files:
                if k.startswith("gfs_"):
                    out.append((from_u8(z[k]), float(k[4:]), obs, mask))
    return out


def batch_crops(pairs, rng, n=16, size=320):
    xs, ys, ms = [], [], []
    while len(xs) < n:
        g, lead, o, m = pairs[rng.integers(len(pairs))]
        i = rng.integers(0, g.shape[0] - size)
        j = rng.integers(0, g.shape[1] - size)
        mc = m[i:i + size, j:j + size]
        if mc.mean() < 0.5:  # need real observations to learn against
            continue
        lat = np.abs(LAT[i:i + size, None]) / 90.0
        x = np.stack([g[i:i + size, j:j + size],
                      np.full((size, size), lead / 48.0, np.float32),
                      np.broadcast_to(lat, (size, size)).astype(np.float32)])
        xs.append(x)
        ys.append(o[i:i + size, j:j + size])
        ms.append(mc)
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
            torch.from_numpy(np.stack(ms)))


def masked_l1(pred_dbz, target, mask, wet_weight=4.0, wet_dbz=15.0):
    w = torch.where(target >= wet_dbz, wet_weight, 1.0) * mask
    return (w * (pred_dbz - target).abs()).sum() / w.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, pairs, dev):
    """Pooled contingency over full held-out fields, corrected vs raw."""
    model.eval()
    counts = {"raw": {t: np.zeros(3) for t in THRESHOLDS},
              "corrected": {t: np.zeros(3) for t in THRESHOLDS}}
    wet = {"raw": 0.0, "corrected": 0.0, "obs": 0.0}
    for g, lead, o, m in pairs:
        h, w = g.shape
        ph, pw = (-h) % POOL_DIVISOR, (-w) % POOL_DIVISOR
        lat = np.broadcast_to(np.abs(LAT[:, None]) / 90.0, g.shape).astype(np.float32)
        x = np.stack([g, np.full_like(g, lead / 48.0), lat])
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
        res = model(torch.from_numpy(x[None]).to(dev))[0, 0, :h, :w].cpu().numpy()
        c = np.clip(g + res, U8_LO, 80.0)
        for name, f in (("raw", g), ("corrected", c)):
            for t in THRESHOLDS:
                counts[name][t] += contingency(f, o, t, m)[:3]
            wet[name] += float((f >= 20.0)[m].sum())
        wet["obs"] += float((o >= 20.0)[m].sum())
    model.train()
    return counts, wet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="ml/gfs_pairs")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=150, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--val-times", type=int, default=10)
    ap.add_argument("--out", default="ml/model/refc_correction_cand.pt",
                    help="a candidate name by default: retrains are gated and "
                         "promoted by ml/retrain_correction.sh, so training "
                         "must never overwrite a shipped checkpoint")
    args = ap.parse_args()

    paths = sorted(Path(args.pairs).glob("*.npz"))
    if len(paths) < 2 * args.val_times:
        raise SystemExit(f"only {len(paths)} valid times in {args.pairs}")
    # Temporal split: newest times held out, so validation is strictly later
    # than training and shares no cycles.
    train_pairs = load_pairs(paths[:-args.val_times])
    val_pairs = load_pairs(paths[-args.val_times:])
    print(f"{len(train_pairs)} train fields, {len(val_pairs)} val fields "
          f"(newest {args.val_times} valid times held out)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RefcUNet(base=args.base, cin=3).to(dev)
    print(f"device {dev}, params "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    counts, wet = evaluate(model, val_pairs, dev)
    base_csi = {t: csi(*counts["raw"][t]) for t in THRESHOLDS}
    print("raw GFS on held-out: "
          + "  ".join(f"{t:g}dBZ {base_csi[t]:.4f}" for t in THRESHOLDS)
          + f"  wet-ratio {wet['raw']/max(wet['obs'],1):.2f}\n")

    best = -1.0
    for ep in range(1, args.epochs + 1):
        run = 0.0
        for _ in range(args.steps):
            x, y, m = batch_crops(train_pairs, rng, args.batch)
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            res = model(x)[:, 0]
            loss = masked_l1(x[:, 0] + res, y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss)
        counts, wet = evaluate(model, val_pairs, dev)
        cs = {t: csi(*counts["corrected"][t]) for t in THRESHOLDS}
        mean_gain = np.mean([cs[t] - base_csi[t] for t in THRESHOLDS])
        flag = ""
        if mean_gain > best:
            best = mean_gain
            torch.save({"state_dict": model.state_dict(), "base": args.base,
                        "cin": 3, "epoch": ep, "mean_gain": best}, args.out)
            flag = "  *saved"
        print(f"epoch {ep:3d}  loss {run/args.steps:.3f}  corrected: "
              + "  ".join(f"{t:g} {cs[t]:.4f}" for t in THRESHOLDS)
              + f"  wet-ratio {wet['corrected']/max(wet['obs'],1):.2f}"
              + f"  mean-gain {mean_gain:+.4f}{flag}")

    print(f"\nbest mean CSI gain over raw GFS: {best:+.4f}")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
