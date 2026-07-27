"""Train the radar-target nowcast model.

    python ml/train_nowcast.py --tiles ml/tiles --epochs 30

Runs on CPU, CUDA or TPU. On Kaggle's free 20 TPU-hours/week this is not close to the
constraint: a 20k-sample, 100-epoch run is well under one hour.

Validation splits by *month*, not at random. Consecutive hours over a 100 km tile are
strongly correlated, so a random split puts near-duplicates of training samples in the
validation set and the score comes out flattering and meaningless.

The baseline to beat is not a published leaderboard, it is what we already measured
against the same radar truth:

    HRRR alone      CSI 0.650 at 1 mm/h   (verify/run_hrrr.py, 45 frames)
    advection       CSI 0.186 at +30 min  (verify/run_radar.py)
    persistence     CSI 0.184 at +30 min

Because the head is zero-initialised the model *starts* at HRRR's score, so any epoch
that does not beat 0.650 is a regression, not slow progress.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nowcast_model import NowcastUNet, csi_torch, weighted_l1  # noqa: E402

U8_LO, U8_HI = -30.0, 60.0
WET_DBZ = 23.0  # 1 mm/h


def from_u8(a):
    return a.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO


class Tiles(Dataset):
    """Loads whole shards into memory; the full set is a few GB of uint8."""

    def __init__(self, paths, months=None, exclude=False):
        xs, ys, hs = [], [], []
        for p in paths:
            with np.load(p) as z:
                if "h" not in z.files:
                    continue
                x, y, h, t = z["x"], z["y"], z["h"], z["t"]
                assert len(x) == len(y) == len(h) == len(t), f"{p} is misaligned"
                if months is not None:
                    m = np.array([s[:7] in months for s in t])
                    if exclude:
                        m = ~m
                    x, y, h = x[m], y[m], h[m]
                if len(x):
                    xs.append(x)
                    ys.append(y)
                    hs.append(h)
        if not xs:
            raise SystemExit("no samples matched; check --tiles and the month split")
        self.x = np.concatenate(xs)
        self.y = np.concatenate(ys)
        self.h = np.concatenate(hs)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (torch.from_numpy(from_u8(self.x[i])),
                torch.from_numpy(from_u8(self.h[i])),
                torch.from_numpy(from_u8(self.y[i])))


def all_months(paths):
    out = set()
    for p in paths:
        with np.load(p) as z:
            if "t" in z.files:
                out.update(s[:7] for s in z["t"])
    return sorted(out)


@torch.no_grad()
def evaluate(model, loader, dev):
    """Mean loss plus CSI for the model and, for reference, HRRR on the same batch."""
    model.eval()
    loss = n = 0.0
    hits = [0.0, 0.0, 0.0]  # model, hrrr, persistence: hits / union accumulators
    union = [0.0, 0.0, 0.0]
    for x, h, y in loader:
        x, h, y = x.to(dev), h.to(dev), y.to(dev)
        pred = model(x, h)
        loss += float(weighted_l1(pred, y)) * len(x)
        n += len(x)
        pers = x[:, -1:].expand_as(y)
        for k, f in enumerate((pred, h, pers)):
            p, t = f >= WET_DBZ, y >= WET_DBZ
            hits[k] += float((p & t).sum())
            union[k] += float((p | t).sum())
    model.train()
    csi = [h_ / u if u else float("nan") for h_, u in zip(hits, union)]
    return loss / max(n, 1), csi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="ml/tiles")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--wet-weight", type=float, default=8.0)
    ap.add_argument("--val-months", type=int, default=6)
    ap.add_argument("--out", default="ml/model/nowcast.pt")
    args = ap.parse_args()

    paths = sorted(Path(args.tiles).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no shards in {args.tiles}; run ml/data/build_tiles.py first")

    months = all_months(paths)
    # Spread the held-out months across the archive rather than taking the newest,
    # so validation covers several seasons instead of one.
    step = max(1, len(months) // max(args.val_months, 1))
    val_months = set(months[::step][:args.val_months])
    print(f"{len(months)} month(s) present; holding out {sorted(val_months)}")

    train = Tiles(paths, val_months, exclude=True)
    val = Tiles(paths, val_months, exclude=False)
    print(f"train {len(train)} samples, val {len(val)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:  # Kaggle TPU, if present
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
    except ImportError:
        pass
    print(f"device: {dev}")

    model = NowcastUNet(base=args.base).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tl = DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True)
    vl = DataLoader(val, batch_size=args.batch)

    loss0, csi0 = evaluate(model, vl, dev)
    print(f"before training: loss {loss0:.3f}  model CSI {csi0[0]:.4f}  "
          f"hrrr {csi0[1]:.4f}  persistence {csi0[2]:.4f}")
    print("(model == hrrr at init by construction; training must improve on it)\n")

    best = -1.0
    for ep in range(1, args.epochs + 1):
        run = n = 0.0
        for x, h, y in tl:
            x, h, y = x.to(dev), h.to(dev), y.to(dev)
            loss = weighted_l1(model(x, h), y, wet_weight=args.wet_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss) * len(x)
            n += len(x)
        vloss, csi = evaluate(model, vl, dev)
        flag = ""
        if csi[0] > best:
            best = csi[0]
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "base": args.base,
                        "val_csi": csi[0]}, args.out)
            flag = "  *saved"
        print(f"epoch {ep:3d}  train {run/max(n,1):.3f}  val {vloss:.3f}  "
              f"CSI {csi[0]:.4f} (hrrr {csi[1]:.4f}){flag}")

    print(f"\nbest val CSI {best:.4f} vs HRRR {csi0[1]:.4f} "
          f"({(best/csi0[1]-1)*100:+.1f}%)")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
