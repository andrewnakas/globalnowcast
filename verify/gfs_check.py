"""Two GFS results were surprising enough to be worth re-testing on their own.

1. Its CSI is nearly flat across lead time. A model with real skill should decay;
   flatness can also mean the score is dominated by climatologically wet regions
   that are easy to "hit" at any lead. Checked by splitting the score by latitude
   band and by how wet each band actually is.

2. Frequency-matching its threshold to the observed wet area *lowers* its CSI.
   Bias correction normally helps, and this single result is the entire reason the
   blend does not bias-correct GFS first, so it deserves more than one data point.

Run: python verify/gfs_check.py   (uses whatever verify/cache/ already holds)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE))

import obs  # noqa: E402
from metrics import contingency, csi  # noqa: E402

CACHE = HERE / "cache"
LAT = obs.GFS_LAT
BANDS = {
    "tropics 0-23": np.abs(LAT) < 23.5,
    "subtrop 23-40": (np.abs(LAT) >= 23.5) & (np.abs(LAT) < 40),
    "midlat 40-60": (np.abs(LAT) >= 40) & (np.abs(LAT) < 60),
    "high 60-70": np.abs(LAT) >= 60,
}


def load(name):
    p = CACHE / f"{name}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def pairs():
    """(gfs field, truth obs, mask, lead_min) for every cached case we can rebuild."""
    out = []
    for gp in sorted(CACHE.glob("gfs_*.npz")):
        stem = gp.stem  # gfs_YYYYMMDDHH_fNNN
        _, cyc, fh = stem.split("_")
        lead_h = int(fh[1:])
        valid = f"{cyc}{'%02d' % 0}"  # cycle hour already in cyc
        # cycle is YYYYMMDDHH; valid = cycle + lead_h, truth frames are :30 past
        from datetime import datetime, timedelta
        c = datetime.strptime(cyc, "%Y%m%d%H")
        v = c + timedelta(hours=lead_h)
        truth = load(f"obs_{v:%Y%m%d%H}30")
        if truth is None:
            continue
        g = load(stem)
        if g is None:
            continue
        out.append((g["dbz"], truth["dbz"], truth["mask"], lead_h * 60))
    return out


def main() -> int:
    cases = pairs()
    if not cases:
        print("no cached GFS/obs pairs; run verify/run.py first")
        return 1
    print(f"{len(cases)} cached GFS/observation pairs\n")

    # --- 1. is the flat CSI an artifact of wet-region dominance? ---
    print("=== GFS CSI by latitude band (dBZ>=20) ===")
    print(f"{'band':<16}{'wet frac':>10}{'CSI':>9}{'bias':>8}   by lead")
    for name, sel in BANDS.items():
        mask2d = np.zeros((LAT.size, obs.GFS_LON.size), bool)
        mask2d[sel] = True
        by_lead = {}
        tot = [0, 0, 0, 0]
        wet_n = wet_d = 0
        for g, t, m, lead in cases:
            mm = m & mask2d
            if not mm.any():
                continue
            c = contingency(g, t, 20.0, mm)
            tot = [a + b for a, b in zip(tot, c)]
            by_lead.setdefault(lead, [0, 0, 0, 0])
            by_lead[lead] = [a + b for a, b in zip(by_lead[lead], c)]
            wet_n += int(((t >= 20) & mm).sum())
            wet_d += int(mm.sum())
        if not wet_d:
            continue
        h, m_, f, _ = tot
        b = (h + f) / (h + m_) if (h + m_) else float("nan")
        leads = "  ".join(f"{l//60}h:{csi(*by_lead[l][:3]):.3f}"
                          for l in sorted(by_lead))
        print(f"{name:<16}{wet_n/wet_d:>10.4f}{csi(h, m_, f):>9.4f}{b:>8.2f}   {leads}")

    print("\nA band that is climatologically wetter should be easier to score in.")
    print("If CSI tracks wet fraction across bands, the flat curve is partly that.\n")

    # --- 2. does frequency matching really hurt? ---
    print("=== frequency-matched vs raw threshold (target: obs wet area @20 dBZ) ===")
    print(f"{'case':<22}{'thr':>7}{'raw CSI':>10}{'matched':>10}{'delta':>9}")
    deltas = []
    for g, t, m, lead in cases:
        target = float(((t >= 20.0) & m).mean())
        if target <= 0:
            continue
        lo, hi = 0.0, 60.0
        for _ in range(40):  # bisect the GFS threshold to match observed wet area
            mid = (lo + hi) / 2
            if float(((g >= mid) & m).mean()) > target:
                lo = mid
            else:
                hi = mid
        raw = csi(*contingency(g, t, 20.0, m)[:3])
        # Same observed truth, but score the model at its frequency-matched threshold.
        matched_pred = np.where(g >= lo, 99.0, obs.FILL)
        mat = csi(*contingency(matched_pred, t, 20.0, m)[:3])
        deltas.append(mat - raw)
        print(f"{'+' + str(lead) + 'm':<22}{lo:>7.2f}{raw:>10.4f}{mat:>10.4f}{mat - raw:>+9.4f}")

    if deltas:
        d = np.array(deltas)
        print(f"\nmean delta {d.mean():+.4f} over {len(d)} cases; "
              f"{int((d < 0).sum())} of {len(d)} got worse when matched.")
        print("Negative means frequency matching hurts, i.e. the model misplaces rain")
        print("rather than merely forecasting too much of it - so rescaling cannot fix it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
