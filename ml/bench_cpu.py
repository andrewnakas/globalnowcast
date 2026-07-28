"""What model size fits the hourly job's CPU budget?

Training happens on a GPU, but the site is built by GitHub Actions on about two cores
with roughly a second per rendered frame. A model that wins on validation CSI and
cannot run there is not an improvement, so this is a shipping gate, not a curiosity.

    python ml/bench_cpu.py                 # global 0.25 deg
    python ml/bench_cpu.py --shape 1301 2951   # CONUS 2 km
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nowcast_model import NowcastUNet, OUT_FRAMES  # noqa: E402

BUDGET_S = 1.0  # per output frame, matching the existing render budget


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=2, type=int, default=[521, 1440],
                    metavar=("H", "W"))
    ap.add_argument("--bases", default="16,24,32,48,64")
    ap.add_argument("--threads", type=int, default=2, help="cores on the runner")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    h, w = args.shape
    print(f"{h}x{w}, {args.threads} thread(s), budget {BUDGET_S:.1f}s per output frame")
    print(f"{'base':>6}{'params':>10}{'total s':>10}{'s/frame':>10}{'verdict':>11}")

    x = torch.randn(1, 4, h, w)
    hr = torch.randn(1, OUT_FRAMES, h, w)
    for base in [int(b) for b in args.bases.split(",")]:
        model = NowcastUNet(base=base).eval()
        n = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            model(x, hr)  # warm up, so the first allocation is not timed
            t = time.time()
            model(x, hr)
            dt = time.time() - t
        per = dt / OUT_FRAMES
        verdict = "ok" if per < BUDGET_S else "too slow"
        print(f"{base:>6}{n / 1e6:>9.2f}M{dt:>10.2f}{per:>10.2f}{verdict:>11}")

    print("\nA tile-wise pass over a region costs roughly the same per pixel, so the "
          "global figure\nis the one that decides what can ship in the hourly job.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
