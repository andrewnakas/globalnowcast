"""Does the model inherit radar's wet bias, or amplify it?

MRMS reports rain over about 2.2x the area gauges measure it (verify/run_asos.py). The
model is trained on MRMS, so it inherits that bias by construction - the question is
whether it makes it worse.

This is the gate in ml/DEPLOY.md. A model that beats HRRR on radar while painting rain
over three times the gauge-measured area is not an improvement for anyone reading the
map, however good its CSI looks.

Compares wet-area fraction for the model, HRRR and the radar truth on the same held-out
tiles. Anything the model adds on top of radar's own bias is its own doing.

    python ml/check_bias.py --ckpt ml/model/nowcast_all32_s1.pt --tiles ml/tiles_all
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "verify"))

from nowcast_model import NowcastUNet, probability_match  # noqa: E402
from train_nowcast import Tiles, all_months, from_u8  # noqa: E402

THRESHOLDS = {"0.2 mm": 18.2, "1 mm": 23.0, "4 mm": 32.6, "8 mm": 37.5}
GAUGE_RATIO = 2.2  # MRMS wet area over gauge wet area, measured in verify/run_asos.py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ml/model/nowcast_all32_s1.pt")
    ap.add_argument("--tiles", default="ml/tiles_all")
    ap.add_argument("--val-months", type=int, default=12)
    ap.add_argument("--max-samples", type=int, default=1500)
    args = ap.parse_args()

    paths = sorted(Path(args.tiles).glob("*.npz"))
    months = all_months(paths)
    step = max(1, len(months) // max(args.val_months, 1))
    vm = set(months[::step][:args.val_months])
    ds = Tiles(paths, vm, exclude=False, limit=args.max_samples)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = NowcastUNet(base=ck.get("base", 32), gated=ck.get("gated", False)).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    x = torch.from_numpy(from_u8(ds.x))
    h = torch.from_numpy(from_u8(ds.h))
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            preds.append(model(x[i:i + 64].to(dev), h[i:i + 64].to(dev)).cpu())
    pred = torch.cat(preds)
    truth = torch.from_numpy(from_u8(ds.y))
    matched = probability_match(pred, truth[:600])

    print(f"{len(truth)} held-out samples\n")
    print("wet-area fraction, and what it implies against gauges")
    print(f"{'threshold':>10}{'radar':>9}{'HRRR':>9}{'model':>9}{'+PM':>9}"
          f"{'model/radar':>13}")
    for label, thr in THRESHOLDS.items():
        r = float((truth >= thr).float().mean())
        hh = float((h >= thr).float().mean())
        m = float((pred >= thr).float().mean())
        p = float((matched >= thr).float().mean())
        ratio = m / r if r else float("nan")
        print(f"{label:>10}{r:>9.4f}{hh:>9.4f}{m:>9.4f}{p:>9.4f}{ratio:>13.2f}")

    r1 = float((truth >= 23.0).float().mean())
    m1 = float((pred >= 23.0).float().mean())
    p1 = float((matched >= 23.0).float().mean())
    print(f"\nAt 1 mm/h the model paints {m1 / r1:.2f}x the radar's wet area.")
    print(f"Radar already runs about {GAUGE_RATIO:.1f}x the gauge-measured area, so the")
    print(f"model would sit near {m1 / r1 * GAUGE_RATIO:.1f}x against gauges, and the")
    print(f"probability-matched output near {p1 / r1 * GAUGE_RATIO:.1f}x.")
    print("\nPM exists to pull that back to the truth distribution. If the matched")
    print("column is close to the radar column, the model is not adding bias of its")
    print("own and what remains is the instrument's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
