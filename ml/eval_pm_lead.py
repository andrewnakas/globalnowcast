"""Is probability matching worth applying per lead, or once across all leads?

PM gave a large gain pooled over leads (+69% at 4 mm/h, +73% at 8 mm/h). But the
model's bias is not constant with lead time - it drifts as the forecast decorrelates -
so a single distribution fitted across all six hours may be correcting the early leads
with the late leads' climatology, or the reverse.

Fitting one distribution per lead is barely more expensive at inference (six small
quantile tables instead of one) so the only question is whether it scores better.

    python ml/eval_pm_lead.py --ckpt ml/model/nowcast_big.pt --tiles ml/tiles_big
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
from nowcast_model import NowcastUNet, probability_match  # noqa: E402
from train_nowcast import Tiles, all_months, from_u8  # noqa: E402

THRESHOLDS = {"1 mm/h": 23.0, "4 mm/h": 32.6, "8 mm/h": 37.5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ml/model/nowcast_big.pt")
    ap.add_argument("--tiles", default="ml/tiles_big")
    ap.add_argument("--val-months", type=int, default=10)
    ap.add_argument("--ref", type=int, default=2000, help="training samples to fit on")
    args = ap.parse_args()

    paths = sorted(Path(args.tiles).glob("*.npz"))
    months = all_months(paths)
    step = max(1, len(months) // max(args.val_months, 1))
    vm = set(months[::step][:args.val_months])
    val = Tiles(paths, vm, exclude=False)
    train = Tiles(paths, vm, exclude=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = NowcastUNet(base=ck.get("base", 32),
                        gated=ck.get("gated", False)).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    x = torch.from_numpy(from_u8(val.x))
    h = torch.from_numpy(from_u8(val.h))
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            preds.append(model(x[i:i + 64].to(dev), h[i:i + 64].to(dev)).cpu())
    pred = torch.cat(preds)
    truth = torch.from_numpy(from_u8(val.y))
    ref = torch.from_numpy(from_u8(train.y[:args.ref]))

    # Two variants: one distribution for everything, or one per lead.
    pooled = probability_match(pred, ref)
    per_lead = torch.stack([probability_match(pred[:, k], ref[:, k])
                            for k in range(pred.shape[1])], dim=1)

    print(f"{len(truth)} held-out samples, fitted on {len(ref)} training samples")
    for label, thr in THRESHOLDS.items():
        print(f"\nCSI at {label}")
        print(f"{'lead':>6}{'model':>9}{'PM pooled':>11}{'PM per-lead':>13}{'HRRR':>9}")
        for k in range(truth.shape[1]):
            t = truth[:, k].numpy()
            m = np.ones_like(t, bool)
            vals = [csi(*contingency(f[:, k].numpy(), t, thr, m)[:3])
                    for f in (pred, pooled, per_lead, h)]
            print(f"{'+' + str(k + 1) + 'h':>6}{vals[0]:>9.4f}{vals[1]:>11.4f}"
                  f"{vals[2]:>13.4f}{vals[3]:>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
