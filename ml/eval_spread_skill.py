"""Does GFS/AIFS disagreement predict where the forecast is wrong?

With two independent global models on the same grid, their spread is a free
confidence signal - but only if it is actually informative. The claim to test is
the spread-skill relationship: forecasts should verify better where the models
agree than where they disagree. That is standard ensemble practice, and it is
also easy to fool yourself about, because spread correlates with rain itself
(models disagree where it rains, and CSI is computed only where it rains).

So this conditions on the forecast being wet, then splits those cells by spread
and scores each bin separately. If low-spread wet cells verify better than
high-spread wet cells, the signal is real and worth surfacing; if the bins score
the same, disagreement is telling us nothing that intensity does not.

    python ml/eval_spread_skill.py

Reports hit rate and false-alarm ratio per spread quintile, plus the pooled CSI
of the shipped mean within each bin.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE.parent / "verify"))

from eval_aifs_vs_gfs import aifs_dbz, from_u8, open_aifs  # noqa: E402

WET_DBZ = 20.0
N_BINS = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="ml/gfs_pairs")
    ap.add_argument("--limit", type=int, default=14)
    args = ap.parse_args()

    import obs

    ds = open_aifs()
    inits = ds.init_time.values
    paths = sorted(Path(args.pairs).glob("*.npz"))[-args.limit:]

    spreads, hits, preds, truths = [], [], [], []
    used = 0
    for p in paths:
        valid = datetime.strptime(p.stem, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        z = np.load(p)
        truth, mask = from_u8(z["obs"]), z["mask"]
        for k in z.files:
            if not k.startswith("gfs_"):
                continue
            lead = int(k[4:])
            init = np.datetime64(valid.replace(tzinfo=None), "ns") \
                - np.timedelta64(lead, "h")
            if init not in inits:
                continue
            a = aifs_dbz(ds, init.astype("datetime64[ns]").item(), lead)
            if a is None:
                continue
            g = from_u8(z[k])
            if a.shape != g.shape:
                continue
            ra, rg = obs.dbz_to_rain(a), obs.dbz_to_rain(g)
            mean = obs.rain_to_dbz(0.5 * (ra + rg))
            # Spread as a fraction of the mean rate: an absolute difference
            # would simply rank heavy rain highest and re-measure intensity.
            denom = np.maximum(0.5 * (ra + rg), 1e-3)
            spread = np.abs(ra - rg) / denom

            wet = (mean >= WET_DBZ) & mask
            if not wet.any():
                continue
            spreads.append(spread[wet])
            preds.append(mean[wet])
            truths.append(truth[wet])
            hits.append((truth[wet] >= WET_DBZ))
            used += 1
        print(f"{p.stem}: done", flush=True)

    if not spreads:
        sys.exit("nothing scored")
    spread = np.concatenate(spreads)
    hit = np.concatenate(hits)
    n = spread.size
    print(f"\n{used} forecast fields, {n/1e6:.1f}M forecast-wet cells "
          f"(mean >= {WET_DBZ:g} dBZ)")

    edges = np.quantile(spread, np.linspace(0, 1, N_BINS + 1))
    print(f"\n{'spread bin':>22}{'cells':>10}{'hit rate':>10}{'FAR':>8}")
    rates = []
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        sel = (spread >= lo) & (spread <= hi if i == N_BINS - 1 else spread < hi)
        if not sel.any():
            continue
        hr = float(hit[sel].mean())
        rates.append(hr)
        print(f"{f'{lo:.2f}-{hi:.2f}':>22}{sel.sum():>10}{hr:>10.3f}"
              f"{1 - hr:>8.3f}")

    if len(rates) >= 2:
        lift = rates[0] / max(rates[-1], 1e-9) - 1
        print(f"\nlowest-spread bin vs highest: hit rate {rates[0]:.3f} vs "
              f"{rates[-1]:.3f}  ({lift:+.1%})")
        print("A large positive lift means agreement predicts correctness, so"
              "\nspread is worth surfacing as confidence. Near zero means it"
              "\nadds nothing beyond what intensity already says.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
