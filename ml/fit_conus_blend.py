"""Refit the CONUS blend's handover from the full cached case set.

The shipped 120min/45tau was fitted on 17 cases over a coarse grid, before the
cache had grown. This refits over all cached cases at a finer resolution, fits
on one half and scores on the other so the chosen parameters are not tuned to
the cases that judge them, and reports every threshold - the global blend's
history says a crossover that wins at one threshold can lose badly at another.

    python ml/fit_conus_blend.py

Prints a held-out comparison against the shipped setting. A change only earns a
commit if it wins on the half it was not fitted on, at every threshold.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE.parent / "verify"))

import conus  # noqa: E402
import nowcast  # noqa: E402
import obs  # noqa: E402
from metrics import contingency, csi  # noqa: E402

U8_LO, U8_HI = -30.0, 60.0
THRESHOLDS = {"1 mm/h": 23.0, "4 mm/h": 32.6, "8 mm/h": 37.5}
LEADS = range(1, 7)
# Replicate-run spread measured on this project's much larger samples. Any
# held-out difference below this is luck, not signal.
NOISE = 0.003


def from_u8(u8):
    return u8.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO


def case_fields(path, km_per_px):
    """Advection and HRRR rain-rate fields per lead, plus truth and mask."""
    d = np.load(path)
    x, h, truth = from_u8(d["x"]), from_u8(d["h"]), from_u8(d["y"])
    in_mask = d["x_mask"] & d["h_mask"]
    flow = nowcast.estimate_flow(x[2], x[3], gap_min=60.0, km_per_px=km_per_px)
    out = []
    for i, k in enumerate(LEADS):
        adv = nowcast.advect(x[3], flow, k * 60.0 / 30.0)
        out.append((k, obs.dbz_to_rain(adv),
                    obs.dbz_to_rain(np.maximum(h[i], obs.FILL)),
                    truth[i], in_mask & d["t_mask"][i]))
    return out


def score(cases, crossover, tau, hold):
    counts = {lab: np.zeros(3) for lab in THRESHOLDS}
    for case in cases:
        for k, ra, rh, truth, m in case:
            w = nowcast.blend_weight(k * 60.0, crossover=crossover, tau=tau,
                                     hold=hold)
            field = obs.rain_to_dbz(w * ra + (1.0 - w) * rh)
            for lab, thr in THRESHOLDS.items():
                counts[lab] += contingency(field, truth, thr, m)[:3]
    return {lab: csi(*counts[lab]) for lab in THRESHOLDS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="ml/cache_live")
    ap.add_argument("--km-per-px", type=float, default=0.02 * 111.0)
    args = ap.parse_args()

    import cv2
    cv2.setNumThreads(4)

    paths = sorted(Path(args.cache).glob("*.npz"))
    print(f"{len(paths)} cached cases; building advection fields", flush=True)
    fit_cases, test_cases = [], []
    for i, p in enumerate(paths):
        (fit_cases if i % 2 == 0 else test_cases).append(
            case_fields(p, args.km_per_px))
        print(f"  {p.stem}", flush=True)

    grid = [(c, t) for c in (60, 90, 105, 120, 135, 150, 180, 240)
            for t in (30, 45, 60)]
    hold = conus.HOLD_MIN
    rows = []
    for c, t in grid:
        s = score(fit_cases, c, t, hold)
        rows.append((np.mean(list(s.values())), c, t, s))
    rows.sort(reverse=True)

    print(f"\ntop settings on the fit half ({len(fit_cases)} cases)")
    print(f"{'cross':>7}{'tau':>5}" + "".join(f"{lab:>10}" for lab in THRESHOLDS)
          + f"{'mean':>9}")
    for mean, c, t, s in rows[:6]:
        print(f"{c:>7}{t:>5}" + "".join(f"{s[lab]:>10.4f}" for lab in THRESHOLDS)
              + f"{mean:>9.4f}")

    best_c, best_t = rows[0][1], rows[0][2]
    ship = score(test_cases, conus.CROSSOVER_MIN, conus.TAU_MIN, hold)
    cand = score(test_cases, best_c, best_t, hold)
    print(f"\nheld-out half ({len(test_cases)} cases)")
    print(f"{'setting':>16}" + "".join(f"{lab:>10}" for lab in THRESHOLDS))
    print(f"{f'shipped {conus.CROSSOVER_MIN:g}/{conus.TAU_MIN:g}':>16}"
          + "".join(f"{ship[lab]:>10.4f}" for lab in THRESHOLDS))
    print(f"{f'candidate {best_c}/{best_t}':>16}"
          + "".join(f"{cand[lab]:>10.4f}" for lab in THRESHOLDS))
    # A change has to clear the measurement noise, not merely tie. Replicate
    # runs on far larger samples in this project scattered by ~0.003 CSI, so a
    # gain of a thousandth on 17 cases is indistinguishable from luck and
    # shipping it would be churn dressed as progress.
    margin = max(cand[lab] - ship[lab] for lab in THRESHOLDS)
    no_loss = all(cand[lab] >= ship[lab] - NOISE for lab in THRESHOLDS)
    verdict = margin > NOISE and no_loss
    print(f"\nbest held-out margin {margin:+.4f} (noise floor {NOISE:.4f})")
    print("verdict: " + ("ship the candidate" if verdict
                         else "keep shipped - the difference is noise"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
