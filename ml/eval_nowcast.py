"""Score a trained nowcast checkpoint on the held-out months, per lead and threshold.

Training reports a single pooled CSI, which hides the thing that actually matters:
skill against HRRR is not uniform across lead time. Recent radar beats the model at
+1 h and loses from +2 h out, so a model that only improved the early leads and a
model that only improved the late ones would print the same headline number.

    python ml/eval_nowcast.py --ckpt ml/model/nowcast.pt
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
from nowcast_model import NowcastUNet  # noqa: E402
from train_nowcast import Tiles, all_months, from_u8  # noqa: E402

THRESHOLDS = {"1 mm/h": 23.0, "2 mm/h": 27.8, "4 mm/h": 32.6, "8 mm/h": 37.5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ml/model/nowcast.pt")
    ap.add_argument("--tiles", default="ml/tiles")
    ap.add_argument("--val-months", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    paths = sorted(Path(args.tiles).glob("*.npz"))
    months = all_months(paths)
    step = max(1, len(months) // max(args.val_months, 1))
    val_months = set(months[::step][:args.val_months])
    ds = Tiles(paths, val_months, exclude=False)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = NowcastUNet(base=ck.get("base", 32), gated=ck.get("gated", False)).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    x = torch.from_numpy(from_u8(ds.x))
    h = torch.from_numpy(from_u8(ds.h))
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), args.batch):
            preds.append(model(x[i:i + args.batch].to(dev),
                               h[i:i + args.batch].to(dev)).cpu().numpy())
    pred = np.concatenate(preds)
    truth = from_u8(ds.y)
    hrrr = from_u8(ds.h)
    pers = np.repeat(from_u8(ds.x)[:, -1:], truth.shape[1], 1)

    print(f"{len(truth)} held-out samples from {sorted(val_months)}")
    print(f"checkpoint {args.ckpt} (base={ck.get('base')})\n")

    print("CSI at 1 mm/h by lead")
    print(f"{'lead':>6}{'model':>9}{'HRRR':>9}{'persist':>9}{'vs HRRR':>9}")
    for k in range(truth.shape[1]):
        m = np.ones_like(truth[:, k], bool)
        a = csi(*contingency(pred[:, k], truth[:, k], 23.0, m)[:3])
        b = csi(*contingency(hrrr[:, k], truth[:, k], 23.0, m)[:3])
        c = csi(*contingency(pers[:, k], truth[:, k], 23.0, m)[:3])
        print(f"{'+' + str(k + 1) + 'h':>6}{a:>9.4f}{b:>9.4f}{c:>9.4f}"
              f"{(a / b - 1) * 100:>+8.1f}%")

    print("\nPooled over all leads")
    print(f"{'threshold':>10}{'model':>9}{'HRRR':>9}{'persist':>9}{'vs HRRR':>9}")
    for label, thr in THRESHOLDS.items():
        m = np.ones_like(truth, bool)
        a = csi(*contingency(pred, truth, thr, m)[:3])
        b = csi(*contingency(hrrr, truth, thr, m)[:3])
        c = csi(*contingency(pers, truth, thr, m)[:3])
        gain = f"{(a / b - 1) * 100:>+8.1f}%" if b else "        -"
        print(f"{label:>10}{a:>9.4f}{b:>9.4f}{c:>9.4f}{gain}")

    # Frequency bias says whether a CSI gain came from finding more rain or from
    # forecasting less of it, which CSI alone cannot distinguish.
    m = np.ones_like(truth, bool)
    for name, f in (("model", pred), ("HRRR", hrrr)):
        hits, miss, fa, _ = contingency(f, truth, 23.0, m)
        print(f"  {name:>6} bias at 1 mm/h: {(hits + fa) / max(hits + miss, 1):.2f}"
              f"  POD {hits / max(hits + miss, 1):.3f}"
              f"  FAR {fa / max(hits + fa, 1):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
