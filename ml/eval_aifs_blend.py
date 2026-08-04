"""Does AIFS improve the blend, and does GFS still earn a place beside it?

Scored standalone on the harvested pairs, AIFS beats GFS by +25/+36/+17% at
5/10/20 dBZ and loses 41% at 30 dBZ - better at ordinary rain, worse at heavy
cores, which is the usual AI-model signature. Two things follow, and neither is
answered by the standalone numbers:

  1. A better model arm is not automatically a better blend. The frame model
     beat HRRR standalone and still lowered the blend, because probability
     matching had correlated its errors with the advection arm's; blending pays
     for decorrelation, not for individual skill.
  2. If AIFS and GFS fail differently by intensity, the max of the two may beat
     either - keeping AIFS's light-rain skill and GFS's heavy cores.

So this scores, on the same cases and the same RRQPE truth: each model alone,
each blended with advection at the shipped handover, and a per-cell combination.

    python ml/eval_aifs_blend.py
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
from metrics import contingency, csi  # noqa: E402

THRESHOLDS = (5.0, 10.0, 20.0, 30.0)
# The blend hands over to the model around here; these pairs are 6-48h leads, so
# every one of them sits well past the crossover and is essentially model-only.
# That is exactly the window a model swap would change.
MODELS = ("gfs", "aifs", "max", "mean")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="ml/gfs_pairs")
    ap.add_argument("--limit", type=int, default=16)
    args = ap.parse_args()

    import obs

    ds = open_aifs()
    inits = ds.init_time.values
    paths = sorted(Path(args.pairs).glob("*.npz"))[-args.limit:]

    counts = {m: {t: np.zeros(3) for t in THRESHOLDS} for m in MODELS}
    wet = {m: 0.0 for m in MODELS}
    wet["obs"] = 0.0
    by_lead = {m: {} for m in MODELS}
    used = 0

    for p in paths:
        valid = datetime.strptime(p.stem, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        z = np.load(p)
        truth, mask = from_u8(z["obs"]), z["mask"]
        scored = False
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
            # Combine in rain rate, never in dBZ: dBZ is logarithmic and a mean
            # of it is a geometric mean that systematically dims the result.
            ra, rg = obs.dbz_to_rain(a), obs.dbz_to_rain(g)
            fields = {"gfs": g, "aifs": a,
                      "max": obs.rain_to_dbz(np.maximum(ra, rg)),
                      "mean": obs.rain_to_dbz(0.5 * (ra + rg))}
            for name, f in fields.items():
                for t in THRESHOLDS:
                    c = contingency(f, truth, t, mask)[:3]
                    counts[name][t] += c
                    by_lead[name].setdefault((lead, t), np.zeros(3))
                    by_lead[name][(lead, t)] += c
                wet[name] += float((f >= 20.0)[mask].sum())
            wet["obs"] += float((truth >= 20.0)[mask].sum())
            scored = True
        used += scored
        print(f"{p.stem}: {'scored' if scored else 'skipped'}", flush=True)

    if not used:
        sys.exit("nothing scored")

    print(f"\n{used} valid times, leads 6-48h, truth = RRQPE")
    print(f"{'':>7}" + "".join(f"{t:>9g}" for t in THRESHOLDS)
          + f"{'wet vs obs':>12}")
    for m in MODELS:
        print(f"{m:>7}" + "".join(f"{csi(*counts[m][t]):>9.4f}" for t in THRESHOLDS)
              + f"{wet[m] / max(wet['obs'], 1.0):>12.2f}")

    base = {t: csi(*counts["gfs"][t]) for t in THRESHOLDS}
    print("\nvs shipped GFS arm:")
    for m in MODELS[1:]:
        gains = [csi(*counts[m][t]) / max(base[t], 1e-9) - 1 for t in THRESHOLDS]
        never_worse = all(g > -0.03 for g in gains)
        print(f"  {m:>5}: " + "  ".join(f"{t:g}dBZ {g:+.1%}"
                                        for t, g in zip(THRESHOLDS, gains))
              + ("   <- wins or ties everywhere" if never_worse else ""))

    leads = sorted({l for l, _ in by_lead["gfs"]})
    print(f"\nCSI at 10 dBZ by lead")
    print(f"{'lead':>6}" + "".join(f"{m:>9}" for m in MODELS))
    for l in leads:
        print(f"{'+' + str(l) + 'h':>6}"
              + "".join(f"{csi(*by_lead[m][(l, 10.0)]):>9.4f}" for m in MODELS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
