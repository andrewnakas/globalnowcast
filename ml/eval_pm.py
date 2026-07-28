"""Does probability matching fix the heavy-rain regression?

The weighted-loss model wins by a wide margin at 1 mm/h and loses at 8 mm/h, because
it paints rain too broadly (frequency bias 1.34 against HRRR's 0.74). Probability
matching keeps the model's spatial pattern but replaces its intensity distribution
with the observed one, so the bias is 1.0 at every threshold by construction.

The calibration distribution is taken from the *training* months, never from the
held-out ones - using validation observations to calibrate would be leakage and would
make the result meaningless.

    python ml/eval_pm.py --ckpt ml/model/nowcast.pt
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

THRESHOLDS = {"1 mm/h": 23.0, "2 mm/h": 27.8, "4 mm/h": 32.6, "8 mm/h": 37.5}


def predict(model, x, h, dev, batch=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(x[i:i + batch].to(dev), h[i:i + batch].to(dev)).cpu())
    return torch.cat(out)


def row(name, pred, truth, mask):
    cells = []
    for thr in THRESHOLDS.values():
        cells.append(csi(*contingency(pred, truth, thr, mask)[:3]))
    hits, miss, fa, _ = contingency(pred, truth, 23.0, mask)
    bias = (hits + fa) / max(hits + miss, 1)
    print(f"{name:>14}" + "".join(f"{c:>10.4f}" for c in cells) + f"{bias:>9.2f}")
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ml/model/nowcast.pt")
    ap.add_argument("--tiles", default="ml/tiles")
    ap.add_argument("--val-months", type=int, default=8)
    args = ap.parse_args()

    paths = sorted(Path(args.tiles).glob("*.npz"))
    months = all_months(paths)
    step = max(1, len(months) // max(args.val_months, 1))
    val_months = set(months[::step][:args.val_months])

    val = Tiles(paths, val_months, exclude=False)
    train = Tiles(paths, val_months, exclude=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = NowcastUNet(base=ck.get("base", 32), gated=ck.get("gated", False)).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    pred = predict(model, torch.from_numpy(from_u8(val.x)),
                   torch.from_numpy(from_u8(val.h)), dev)
    truth = torch.from_numpy(from_u8(val.y))
    hrrr = torch.from_numpy(from_u8(val.h))

    # Calibrate on training observations only.
    ref = torch.from_numpy(from_u8(train.y[:2000]))
    matched = probability_match(pred, ref)

    t, m = truth.numpy(), np.ones_like(truth.numpy(), bool)
    print(f"{len(t)} held-out samples, calibrated on {len(ref)} training samples\n")
    print(f"{'':>14}" + "".join(f"{k:>10}" for k in THRESHOLDS) + f"{'bias':>9}")
    a = row("HRRR", hrrr.numpy(), t, m)
    b = row("model", pred.numpy(), t, m)
    c = row("model + PM", matched.numpy(), t, m)

    print("\nvs HRRR:")
    for name, vals in (("model", b), ("model + PM", c)):
        deltas = "  ".join(f"{k}: {(v / h - 1) * 100:+.1f}%"
                           for k, v, h in zip(THRESHOLDS, vals, a))
        print(f"  {name:>10}  {deltas}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
