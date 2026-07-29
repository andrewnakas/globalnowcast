"""Refit the probability-matching tables on full-frame climatology.

The original tables were fitted on training tiles, and tiles are *selected to be
wet* (build_tiles keeps sequences with >=5% wet targets). Applied to a full CONUS
frame - which is a few percent wet - rank-based PM forces the frame to the tiles'
wet fraction and paints ~10x the observed rain area. Measured, not theoretical:
the serving smoke test produced 21-25% wet against 2.5% observed.

PM's reference must be the climatology of the domain it is applied to. This fits
per-lead quantile tables from the full-frame MRMS truth in the ml/eval_live_inputs
cache, using only every second case (by date order) so the remaining half can score
the result without the reference having seen its truth.

Caveat, on purpose: the cache spans ~2 weeks of one season. The weekly verify job
is the long-term fix (Phase 3 refits as the archive grows seasons).

    python ml/fit_pm_full.py --cache ml/cache_live --out ml/model/pm_tables.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

U8_LO, U8_HI = -30.0, 60.0
N_Q = 2048


def from_u8(u8):
    return u8.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="ml/cache_live")
    ap.add_argument("--out", default="ml/model/pm_tables.npz")
    ap.add_argument("--half", choices=["even", "odd", "all"], default="even",
                    help="which cases to fit on; score on the others")
    args = ap.parse_args()

    paths = sorted(Path(args.cache).glob("*.npz"))
    if args.half != "all":
        keep = 0 if args.half == "even" else 1
        paths = [p for i, p in enumerate(paths) if i % 2 == keep]
    if len(paths) < 5:
        sys.exit(f"only {len(paths)} cases in {args.cache} - not enough to fit on")

    grid = np.linspace(0.0, 1.0, N_Q)
    leads = None
    acc = None
    for p in paths:
        d = np.load(p)
        y = from_u8(d["y"])          # (6, H, W) truth
        m = d["x_mask"] & d["h_mask"]
        if leads is None:
            leads = y.shape[0]
            acc = [[] for _ in range(leads)]
        for k in range(leads):
            mk = m & d["t_mask"][k]
            # Per-case quantiles, averaged across cases: same memory footprint per
            # case, and the average of quantile curves is a fine estimator here.
            acc[k].append(np.quantile(y[k][mk], grid))

    tables = np.stack([np.mean(a, axis=0) for a in acc]).astype(np.float32)
    np.savez(args.out, ref_q=tables, n=N_Q, fitted_on=len(paths),
             source=f"full-frame truth, {args.half} half of {args.cache}")
    wet = [(t >= 23.0).mean() for t in tables]
    print(f"wrote {args.out}: ref_q {tables.shape} from {len(paths)} cases "
          f"({args.half} half)")
    print(f"reference wet fraction (>=23 dBZ) per lead: "
          + " ".join(f"{w:.4f}" for w in wet))
    return 0


if __name__ == "__main__":
    sys.exit(main())
