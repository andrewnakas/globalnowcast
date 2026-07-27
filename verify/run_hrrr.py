"""Pool HRRR-vs-radar skill over many cases, seasons and regions.

A single wet Iowa case put HRRR three times ahead of the satellite retrieval against
radar truth. That gap is far too large to be noise, but the exact numbers are worth
pooling before any of this is quoted or a training set is built around it.

Samples chunk-aligned MRMS tiles spread over the archive, keeps the frames with real
rain in them (a dry tile scores nothing interesting and would just dilute the pool),
and scores HRRR composite reflectivity against MRMS on the identical grid and hour.

    python verify/run_hrrr.py --tiles 12 --frames 6
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ml" / "data"))
sys.path.insert(0, str(HERE))

import dynamical as dyn  # noqa: E402
from metrics import contingency, csi  # noqa: E402

# Marshall-Palmer, matching the rest of the project: 1/2/4/8 mm/h.
THRESHOLDS = {1.0: 23.0, 2.0: 27.8, 4.0: 32.6, 8.0: 37.5}
MIN_WET = 0.02  # a frame needs this much rain to be worth scoring


def sample_tiles(n_tiles: int, seed: int = 0):
    """Chunk-aligned (i, j, t0) triples spread across CONUS and the archive.

    Latitude is restricted away from the domain edges, where the mosaic thins out and
    'no coverage' would otherwise masquerade as dry.
    """
    nt, ny, nx = dyn.grid_size()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_tiles):
        i = int(rng.integers(4, (ny - dyn.TILE) // dyn.TILE - 4)) * dyn.TILE
        j = int(rng.integers(4, (nx - dyn.TILE) // dyn.TILE - 4)) * dyn.TILE
        t0 = int(rng.integers(0, (nt - 648) // 648)) * 648
        out.append((i, j, t0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=12)
    ap.add_argument("--frames", type=int, default=6, help="wet frames scored per tile")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pooled = defaultdict(lambda: [0, 0, 0, 0])
    by_season = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    n_frames = 0
    intensity = []

    for k, (i, j, t0) in enumerate(sample_tiles(args.tiles, args.seed), 1):
        try:
            tile = dyn.mrms_tile(i, j, t0)
        except Exception as e:  # noqa: BLE001
            print(f"[{k}] tile ({i},{j},{t0}) failed: {e}")
            continue

        wet = tile.wet_fraction
        idx = np.argsort(-wet)[:args.frames]
        idx = [int(x) for x in idx if wet[x] >= MIN_WET]
        if not idx:
            print(f"[{k}] tile ({i},{j}) dry, skipping")
            continue

        hrrr = dyn.hrrr_on(tile.lat, tile.lon, tile.times[idx])
        if hrrr is None:
            print(f"[{k}] tile ({i},{j}) outside HRRR domain")
            continue

        for m, t_idx in enumerate(idx):
            truth = tile.dbz[t_idx]
            pred = hrrr[m]
            mask = np.ones_like(truth, bool)
            month = int(str(tile.times[t_idx])[5:7])
            season = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                      6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON",
                      11: "SON"}[month]
            for mmhr, dbz in THRESHOLDS.items():
                c = contingency(pred, truth, dbz, mask)
                pooled[mmhr] = [a + b for a, b in zip(pooled[mmhr], c)]
                by_season[season][mmhr] = [a + b for a, b in
                                           zip(by_season[season][mmhr], c)]
            intensity.append((float(pred.max()), float(truth.max())))
            n_frames += 1
        print(f"[{k}] tile ({i:4d},{j:4d}) {str(tile.times[idx[0]])[:10]}  "
              f"{len(idx)} frame(s), wettest {wet[idx[0]]:.2f}")

    if not n_frames:
        print("nothing scored")
        return 1

    print(f"\n{n_frames} frames from {args.tiles} tiles across the archive")
    print("\n=== HRRR composite reflectivity vs MRMS radar (same grid, same hour) ===")
    print(f"{'threshold':>12}{'CSI':>9}{'POD':>8}{'FAR':>8}{'bias':>8}")
    for mmhr in THRESHOLDS:
        h, m, f, _ = pooled[mmhr]
        print(f"{f'{mmhr:g} mm/h':>12}{csi(h, m, f):>9.4f}"
              f"{h / max(h + m, 1):>8.3f}{f / max(h + f, 1):>8.3f}"
              f"{(h + f) / max(h + m, 1):>8.2f}")

    print("\n=== by season, CSI ===")
    print(f"{'season':>8}" + "".join(f"{f'{t:g}mm':>9}" for t in THRESHOLDS))
    for s in sorted(by_season):
        row = "".join(f"{csi(*by_season[s][t][:3]):>9.4f}" for t in THRESHOLDS)
        print(f"{s:>8}{row}")

    if intensity:
        p = np.array([x[0] for x in intensity])
        t = np.array([x[1] for x in intensity])
        print(f"\npeak dBZ per frame: HRRR {p.mean():.1f} vs radar {t.mean():.1f} "
              f"({p.mean() - t.mean():+.1f} dB)")
        print("A positive offset means the model over-forecasts intensity, which is")
        print("exactly the systematic error a trained correction can absorb.")

    print("\nFor comparison, the satellite retrieval against radar at zero lead scored")
    print("CSI 0.182 / 0.105 / 0.080 at 1 / 4 / 8 mm/h (verify/BENCHMARKS.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
