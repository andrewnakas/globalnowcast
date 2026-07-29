"""Which probability-matching scheme survives contact with a full frame?

Global rank-based PM was measured to *hurt* full-frame CSI (0.138 -> 0.106 at
1 mm/h) even with correctly-fitted climatology: a CONUS frame is ~98.5% dry, and
rank noise among millions of dry cells competes with real drizzle for the wet end
of the reference. On wet-selected tiles this never showed.

Candidates, scored on the cached live cases:
  raw          no PM at all
  global       rank-match the whole covered frame (the shipped-tile scheme)
  active-T     rank-match only cells with prediction >= T dBZ, against the
               matching upper tail of the reference; dry cells pass through
               untouched, so PM can no longer invent rain where there is none

    python ml/eval_pm_variants.py --half odd

Uses the same cache and masks as ml/eval_live_inputs.py; runs in minutes.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "verify"))

from export_pm import pm_apply  # noqa: E402
from metrics import contingency, csi  # noqa: E402

U8_LO, U8_HI = -30.0, 60.0
THRESHOLDS = {"1 mm/h": 23.0, "4 mm/h": 32.6, "8 mm/h": 37.5}


def from_u8(u8):
    return u8.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO


def pm_active(pred, ref_q, thresh):
    """PM restricted to the active set: cells at/above `thresh` keep their ranks
    and take values from the reference's matching upper tail. Everything below
    passes through, so a dry cell stays exactly as predicted."""
    act = pred >= thresh
    n_act = int(act.sum())
    if n_act == 0:
        return pred
    lo = int(round((1.0 - n_act / pred.size) * (ref_q.size - 1)))
    out = pred.copy()
    out[act] = pm_apply(pred[act], ref_q[lo:])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="ml/cache_live")
    ap.add_argument("--onnx", default="ml/model/nowcast.onnx")
    ap.add_argument("--pm", default="ml/model/pm_tables.npz")
    ap.add_argument("--half", choices=["even", "odd", "all"], default="odd")
    ap.add_argument("--active", default="5,10,15", help="thresholds to try, dBZ")
    args = ap.parse_args()

    import onnxruntime as ort

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    tables = np.load(args.pm)["ref_q"]
    paths = sorted(Path(args.cache).glob("*.npz"))
    if args.half != "all":
        keep = 0 if args.half == "even" else 1
        paths = [p for i, p in enumerate(paths) if i % 2 == keep]

    variants = ["raw", "global"] + [f"active-{t}" for t in
                (int(x) for x in args.active.split(","))]
    leads = range(1, 7)
    counts = {v: {lab: np.zeros(3) for lab in THRESHOLDS} for v in variants}
    wet = {v: 0.0 for v in variants}
    wet_truth = 0.0

    for p in paths:
        d = np.load(p)
        x, h = from_u8(d["x"]), from_u8(d["h"])
        truth = from_u8(d["y"])
        pred = sess.run(["forecast"], {"radar": x[None], "hrrr": h[None]})[0][0]
        in_mask = d["x_mask"] & d["h_mask"]
        for i, k in enumerate(leads):
            m = in_mask & d["t_mask"][i]
            pv = pred[i][m]
            fields = {"raw": pv, "global": pm_apply(pv, tables[i])}
            for v in variants[2:]:
                t = float(v.split("-")[1])
                fields[v] = pm_active(pv, tables[i], t)
            tv = truth[i][m]
            ones = np.ones_like(tv, bool)
            for v, f in fields.items():
                for lab, thr in THRESHOLDS.items():
                    counts[v][lab] += contingency(f, tv, thr, ones)[:3]
                wet[v] += float((f >= 23.0).sum())
            wet_truth += float((tv >= 23.0).sum())
        print(f"{p.stem}: done", flush=True)

    print(f"\n{len(paths)} cases ({args.half} half), pooled over +1..+6h")
    print(f"{'variant':>12}" + "".join(f"{lab:>10}" for lab in THRESHOLDS)
          + f"{'wet ratio':>11}")
    for v in variants:
        row = "".join(f"{csi(*counts[v][lab]):>10.4f}" for lab in THRESHOLDS)
        print(f"{v:>12}{row}{wet[v] / max(wet_truth, 1.0):>11.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
