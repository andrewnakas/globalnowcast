"""Train the CONUS model on the mixed wet/dry frame diet, with an anti-blur loss.

Two measured facts drive this file. First, every wet-curated model collapsed at
frame scale because it never saw dry sky (ml/tiles_frame includes it). Second, the
RAPSD diagnostic shows the trained models carry HALF the truth's small-scale power
(0.23 vs 0.50 high-frequency fraction) while HRRR keeps 0.47 - L1 training blurs,
PM then repaints a sharp histogram onto blurred shapes, and CSI pays for both.
The standard cheap counter from the nowcasting literature is a gradient-matching
term: L1 on the spatial gradients alongside L1 on the values.

    python ml/train_frame.py --tiles ml/tiles_frame --grad-weight 1.0
    python ml/train_frame.py --tiles ml/tiles_frame --grad-weight 0.0  # control

Run both: the gradient term's worth is a claim to measure, not assume.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nowcast_model import NowcastUNet, weighted_l1  # noqa: E402
from train_nowcast import Tiles, _save, all_months, from_u8  # noqa: E402

WET_DBZ = 23.0


def grad_l1(pred, target):
    """L1 on horizontal+vertical differences: penalises smoothing directly."""
    dy = (pred[..., 1:, :] - pred[..., :-1, :]) - \
         (target[..., 1:, :] - target[..., :-1, :])
    dx = (pred[..., :, 1:] - pred[..., :, :-1]) - \
         (target[..., :, 1:] - target[..., :, :-1])
    return dy.abs().mean() + dx.abs().mean()


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    hits = [0.0, 0.0]
    union = [0.0, 0.0]
    sharp = [0.0, 0.0, 0.0]  # mean |grad| of pred, hrrr, truth: a blur proxy
    n = 0
    for x, h, y in loader:
        x, h, y = x.to(dev), h.to(dev), y.to(dev)
        pred = model(x, h)
        for k, f in enumerate((pred, h)):
            p, t = f >= WET_DBZ, y >= WET_DBZ
            hits[k] += float((p & t).sum())
            union[k] += float((p | t).sum())
        for k, f in enumerate((pred, h, y)):
            sharp[k] += float((f[..., 1:, :] - f[..., :-1, :]).abs().mean()) * len(x)
        n += len(x)
    model.train()
    return ([h_ / u if u else float("nan") for h_, u in zip(hits, union)],
            [s / max(n, 1) for s in sharp])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="ml/tiles_frame")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--wet-weight", type=float, default=8.0)
    ap.add_argument("--grad-weight", type=float, default=1.0)
    ap.add_argument("--val-months", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"ml/model/nowcast_frame_g{args.grad_weight:g}.pt"

    paths = sorted(Path(args.tiles).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no shards in {args.tiles}")
    months = all_months(paths)
    step = max(1, len(months) // max(args.val_months, 1))
    vm = set(months[::step][:args.val_months])
    print(f"{len(months)} month(s); holding out {sorted(vm)}")
    train = Tiles(paths, vm, exclude=True)
    val = Tiles(paths, vm, exclude=False)
    print(f"train {len(train)}, val {len(val)}  (mixed wet/dry diet)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = NowcastUNet(base=args.base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    from torch.utils.data import DataLoader
    tl = DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True)
    vl = DataLoader(val, batch_size=args.batch)

    csi0, sharp0 = evaluate(model, vl, dev)
    print(f"init: CSI {csi0[0]:.4f} (hrrr {csi0[1]:.4f})  "
          f"|grad| pred {sharp0[0]:.3f} hrrr {sharp0[1]:.3f} truth {sharp0[2]:.3f}\n")

    best = -1.0
    for ep in range(1, args.epochs + 1):
        run = n = 0.0
        for x, h, y in tl:
            x, h, y = x.to(dev), h.to(dev), y.to(dev)
            pred = model(x, h)
            loss = weighted_l1(pred, y, wet_weight=args.wet_weight) \
                + args.grad_weight * grad_l1(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss) * len(x)
            n += len(x)
        csi, sharp = evaluate(model, vl, dev)
        flag = ""
        if csi[0] > best:
            best = csi[0]
            _save({"state_dict": model.state_dict(), "base": args.base,
                   "gated": False, "val_csi": best, "epoch": ep,
                   "grad_weight": args.grad_weight}, out)
            flag = "  *saved"
        print(f"epoch {ep:3d}  loss {run/max(n,1):.3f}  CSI {csi[0]:.4f} "
              f"(hrrr {csi[1]:.4f})  |grad| {sharp[0]:.3f}"
              f"/{sharp[2]:.3f}{flag}", flush=True)

    print(f"\nbest val CSI {best:.4f}  saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
