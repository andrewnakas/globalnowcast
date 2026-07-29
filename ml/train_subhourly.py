"""Train the 10-minute-cadence radar model, in the published-benchmark format.

    python ml/train_subhourly.py --tiles ml/tiles_10min --epochs 60

Same skeleton as ml/train_nowcast.py with three deliberate differences: the input
is radar only (4 frames, 40 min of history - the DGMR/SEVIR formulation, no NWP
side input), the output is 18 frames (3 h at 10-min steps), and the reference the
model starts from is persistence, since without an HRRR channel the identity-init
head reproduces the last input frame. Published nowcasting work reports +30 to
+90 min, so validation prints those leads specifically - this model is the only
honest bridge between our numbers and that literature (see verify/BENCHMARKS.md).

Persistence at 10-min cadence decays 0.537 / 0.356 / 0.268 / 0.219 at
+10/+30/+60/+90 min (measured, [[nowcast-ceilings]]), so the bar moves with lead.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nowcast_model import NowcastUNet, weighted_l1  # noqa: E402
from train_nowcast import _save, all_months, from_u8  # noqa: E402

OUT_FRAMES = 18
WET_DBZ = 23.0
REPORT_LEADS = {"+30m": 2, "+60m": 5, "+90m": 8, "+180m": 17}  # index into outputs


class SubTiles(Dataset):
    """Radar-only shards: x (N,4,H,W), y (N,18,H,W), t (N)."""

    def __init__(self, paths, months=None, exclude=False):
        xs, ys = [], []
        for p in paths:
            with np.load(p) as z:
                t = z["t"]
                if months is not None:
                    m = np.array([s[:7] in months for s in t])
                    if exclude:
                        m = ~m
                    if not m.any():
                        continue
                else:
                    m = slice(None)
                x, y = z["x"][m], z["y"][m]
                if len(x):
                    xs.append(x)
                    ys.append(y)
        if not xs:
            raise SystemExit("no samples matched; check --tiles and the month split")
        self.x = np.concatenate(xs)
        self.y = np.concatenate(ys)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (torch.from_numpy(from_u8(self.x[i])),
                torch.from_numpy(from_u8(self.y[i])))


@torch.no_grad()
def evaluate(model, loader, dev):
    """Loss, pooled CSI, and per-report-lead CSI for model and persistence."""
    model.eval()
    loss = n = 0.0
    acc = {name: [0.0, 0.0, 0.0, 0.0] for name in ("all",) + tuple(REPORT_LEADS)}
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        pred = model(x)
        loss += float(weighted_l1(pred, y)) * len(x)
        n += len(x)
        pers = x[:, -1:].expand_as(y)
        for name, k in [("all", slice(None))] + \
                [(nm, sl) for nm, sl in REPORT_LEADS.items()]:
            for j, f in enumerate((pred, pers)):
                p = f[:, k] >= WET_DBZ
                t = y[:, k] >= WET_DBZ
                acc[name][2 * j] += float((p & t).sum())
                acc[name][2 * j + 1] += float((p | t).sum())
    model.train()
    out = {}
    for name, (h1, u1, h2, u2) in acc.items():
        out[name] = (h1 / u1 if u1 else float("nan"),
                     h2 / u2 if u2 else float("nan"))
    return loss / max(n, 1), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="ml/tiles_10min")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--wet-weight", type=float, default=8.0)
    ap.add_argument("--val-months", type=int, default=4)
    ap.add_argument("--out", default="ml/model/nowcast_10min.pt")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    paths = sorted(Path(args.tiles).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no shards in {args.tiles}; run build_subhourly.py first")

    months = all_months(paths)
    step = max(1, len(months) // max(args.val_months, 1))
    val_months = set(months[::step][:args.val_months])
    print(f"{len(months)} month(s) present; holding out {sorted(val_months)}")

    train = SubTiles(paths, val_months, exclude=True)
    val = SubTiles(paths, val_months, exclude=False)
    print(f"train {len(train)} samples, val {len(val)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    model = NowcastUNet(base=args.base, out_frames=OUT_FRAMES,
                        use_hrrr=False).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tl = DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True)
    vl = DataLoader(val, batch_size=args.batch)

    loss0, csi0 = evaluate(model, vl, dev)
    lead_str = "  ".join(f"{nm} {csi0[nm][1]:.3f}" for nm in REPORT_LEADS)
    print(f"before training: loss {loss0:.3f}  (persistence by lead: {lead_str})")
    print("(model == persistence at init by construction)\n")

    best = -1.0
    start_ep = 1
    last_path = Path(str(args.out) + ".last")
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        best = ck.get("best", -1.0)
        start_ep = ck["epoch"] + 1
        print(f"resumed from epoch {ck['epoch']} (best CSI {best:.4f})\n")

    for ep in range(start_ep, args.epochs + 1):
        run = n = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            loss = weighted_l1(model(x), y, wet_weight=args.wet_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss) * len(x)
            n += len(x)
        vloss, csi = evaluate(model, vl, dev)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        flag = ""
        if csi["all"][0] > best:
            best = csi["all"][0]
            _save({"state_dict": model.state_dict(), "base": args.base,
                   "out_frames": OUT_FRAMES, "use_hrrr": False,
                   "val_csi": best, "epoch": ep}, args.out)
            flag = "  *saved"
        _save({"state_dict": model.state_dict(), "base": args.base,
               "optimizer": opt.state_dict(), "epoch": ep, "best": best},
              last_path)
        leads = "  ".join(f"{nm} {csi[nm][0]:.3f}/{csi[nm][1]:.3f}"
                          for nm in REPORT_LEADS)
        print(f"epoch {ep:3d}  train {run/max(n,1):.3f}  val {vloss:.3f}  "
              f"CSI {csi['all'][0]:.4f} (pers {csi['all'][1]:.4f})  "
              f"[model/pers: {leads}]{flag}")

    print(f"\nbest pooled val CSI {best:.4f}")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
