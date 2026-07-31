"""Does the frame-diet model improve the shipped blend when it replaces HRRR?

The model beats HRRR on live frames (+4-5% pooled at 1 mm/h) but loses to the
shipped blend, which is MRMS advection handed over to HRRR around +2 h. Those
facts are compatible: advection owns the early leads either way, and the model
only has to beat HRRR where the blend is already leaning on HRRR. So the
question is not "model vs blend" but "does swapping the blend's NWP arm for the
model raise the blend".

Scored on the same cached live cases as every other gate, so the numbers sit
directly beside ml/out_live_eval*.json.

    python ml/eval_blend_variants.py --onnx ml/model/nowcast_frame_g0.onnx
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
from export_pm import pm_apply  # noqa: E402
from metrics import contingency, csi  # noqa: E402

U8_LO, U8_HI = -30.0, 60.0
THRESHOLDS = {"1 mm/h": 23.0, "4 mm/h": 32.6, "8 mm/h": 37.5}
LEADS = range(1, 7)


def from_u8(u8):
    return u8.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="ml/model/nowcast_frame_g0.onnx")
    ap.add_argument("--pm", default="ml/model/pm_tables.npz")
    ap.add_argument("--cache", default="ml/cache_live")
    ap.add_argument("--half", choices=["even", "odd", "all"], default="odd")
    ap.add_argument("--km-per-px", type=float, default=0.02 * 111.0)
    args = ap.parse_args()

    import cv2
    import onnxruntime as ort

    cv2.setNumThreads(4)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    tables = np.load(args.pm)["ref_q"]

    paths = sorted(Path(args.cache).glob("*.npz"))
    if args.half != "all":
        keep = 0 if args.half == "even" else 1
        paths = [p for i, p in enumerate(paths) if i % 2 == keep]

    variants = ("advection", "hrrr", "model_pm", "blend_hrrr", "blend_model")
    counts = {v: {lab: {k: np.zeros(3) for k in LEADS} for lab in THRESHOLDS}
              for v in variants}

    for p in paths:
        d = np.load(p)
        x, h, truth = from_u8(d["x"]), from_u8(d["h"]), from_u8(d["y"])
        in_mask = d["x_mask"] & d["h_mask"]
        pred = sess.run(["forecast"], {"radar": x[None], "hrrr": h[None]})[0][0]
        pm = np.full_like(pred, U8_LO)
        for i in range(pred.shape[0]):
            pm[i][in_mask] = pm_apply(pred[i][in_mask], tables[i])

        # Flow from the last two hourly frames, exactly as pipeline/conus.py
        # would have had available at the anchor.
        flow = nowcast.estimate_flow(x[2], x[3], gap_min=60.0,
                                     km_per_px=args.km_per_px)
        for i, k in enumerate(LEADS):
            m = in_mask & d["t_mask"][i]
            minutes = k * 60.0
            adv = nowcast.advect(x[3], flow, minutes / 30.0)
            w = nowcast.blend_weight(minutes, crossover=conus.CROSSOVER_MIN,
                                     tau=conus.TAU_MIN, hold=conus.HOLD_MIN)
            ra = obs.dbz_to_rain(adv)
            fields = {"advection": adv, "hrrr": h[i], "model_pm": pm[i]}
            for name, nwp in (("blend_hrrr", h[i]), ("blend_model", pm[i])):
                rate = w * ra + (1.0 - w) * obs.dbz_to_rain(
                    np.maximum(nwp, obs.FILL))
                fields[name] = obs.rain_to_dbz(rate)
            for name, f in fields.items():
                for lab, thr in THRESHOLDS.items():
                    counts[name][lab][k] += contingency(f, truth[i], thr, m)[:3]
        print(f"{p.stem}: done", flush=True)

    print(f"\n{len(paths)} cases, {Path(args.onnx).name}, "
          f"crossover {conus.CROSSOVER_MIN:g}min")
    for lab in THRESHOLDS:
        print(f"\nCSI at {lab}")
        print(f"{'lead':>6}" + "".join(f"{v:>13}" for v in variants))
        for k in LEADS:
            print(f"{'+' + str(k) + 'h':>6}"
                  + "".join(f"{csi(*counts[v][lab][k]):>13.4f}" for v in variants))
        pooled = {v: csi(*sum(counts[v][lab][k] for k in LEADS)) for v in variants}
        print(f"{'all':>6}" + "".join(f"{pooled[v]:>13.4f}" for v in variants))
        if lab == "1 mm/h":
            gain = pooled["blend_model"] / max(pooled["blend_hrrr"], 1e-9) - 1
            print(f"\nblend_model vs blend_hrrr at 1 mm/h: {gain:+.1%}"
                  f"  ({'SHIP IT' if gain > 0.02 else 'not worth the plumbing'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
