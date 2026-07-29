"""The CONUS radar-model layer: MRMS history + HRRR forecast -> 6 corrected hours.

Serving side of ml/nowcast_model.py, structured exactly like pipeline/correct.py: a
lazy, thread-safe load that disables itself unless every artifact is present, a
NOWCAST_RADAR=0 kill switch, and no path that can raise into the hourly build.

Two artifacts are required, and the layer stays off unless BOTH exist:
  ml/model/nowcast.onnx     the trained sequence model
  ml/model/pm_tables.npz    per-lead probability-matching quantiles

PM is a shipping requirement, not a refinement: the raw model over-paints light rain
1.53x and produces only 41% of the heavy rain present (ml/check_bias.py). Shipping
ONNX output without the tables would be worse than shipping nothing, so a missing
table file disables the layer rather than degrading it.
"""
import os
import sys
import threading
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hrrr  # noqa: E402
import mrms  # noqa: E402
import obs  # noqa: E402

_MODEL_DIR = Path(__file__).resolve().parent.parent / "ml" / "model"
MODEL_PATH = _MODEL_DIR / "nowcast.onnx"
PM_PATH = _MODEL_DIR / "pm_tables.npz"

_state = None  # None = not tried yet; (session, tables) = ready; False = disabled
_lock = threading.Lock()


def _load():
    global _state
    if _state is not None:
        return _state or None
    with _lock:
        if _state is not None:
            return _state or None
        loaded = False
        if (os.environ.get("NOWCAST_RADAR", "1") != "0"
                and MODEL_PATH.exists() and PM_PATH.exists()):
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                # Unlike correct.py this runs after the frame pool has drained, so
                # it may use a couple of threads without oversubscribing.
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 1
                session = ort.InferenceSession(str(MODEL_PATH), sess_options=opts,
                                               providers=["CPUExecutionProvider"])
                tables = np.load(PM_PATH)["ref_q"].astype(np.float32)
                loaded = (session, tables)
                print(f"radar model: using {MODEL_PATH.name} + {PM_PATH.name}")
            except Exception as e:  # noqa: BLE001
                print(f"radar model: disabled ({e})")
                loaded = False
        _state = loaded
    return _state or None


def is_active() -> bool:
    return _load() is not None


def pm_apply(pred: np.ndarray, ref_q: np.ndarray) -> np.ndarray:
    """Probability matching, ported from ml/nowcast_model.probability_match and
    verified against it in ml/export_pm.py. Each cell keeps its rank; the values
    come from the reference distribution. Apply to one full lead's field (or its
    covered subset), never tile-by-tile."""
    flat = pred.reshape(-1)
    order = np.argsort(np.argsort(flat, kind="stable"), kind="stable")
    idx = np.linspace(0, ref_q.size - 1, flat.size).astype(np.int64)
    ref = ref_q[np.clip(idx, 0, ref_q.size - 1)]
    return ref[order].reshape(pred.shape)


def predict(session, now, grid=None):
    """Run the full layer for the live build: latest anchor, then predict_anchor."""
    model = _load()
    if model is None:
        return None
    anchor = mrms.latest_anchor(session, now)
    if anchor is None:
        print("radar model: no recent MRMS anchor", file=sys.stderr)
        return None
    return predict_anchor(session, anchor, grid)


def predict_anchor(session, anchor, grid=None, avail=None, keep_hrrr=False):
    """Fetch inputs anchored at a given top-of-hour, infer, probability-match.

    `avail` simulates a historical build time for the HRRR cycle walk-back
    (verification uses anchor+10min, matching the hourly job's schedule).

    Returns None on any failure, or a dict with:
        anchor      the top-of-hour the forecast is anchored to (datetime)
        by_valid    {valid time: dBZ field} - the anchor's observed radar at lead 0
                    plus the six probability-matched forecast hours
        mask        bool array, where the layer has radar+HRRR support
        cycle       the HRRR cycle used (for the manifest)
    """
    model = _load()
    if model is None:
        return None
    ort_sess, tables = model
    grid = grid or obs.CONUS_2KM
    try:
        hist = mrms.fetch_history(session, anchor, grid)
        if hist is None:
            print("radar model: MRMS history incomplete", file=sys.stderr)
            return None
        x, x_mask, _ = hist
        fc = hrrr.fetch_forecast(session, anchor, grid, avail=avail)
        if fc is None:
            return None
        h, h_mask, cycle = fc

        pred = ort_sess.run(["forecast"], {"radar": x[None], "hrrr": h[None]})[0][0]
        mask = x_mask & h_mask
        by_valid = {anchor: x[-1]}
        for i in range(pred.shape[0]):
            f = np.full_like(pred[i], obs.FILL)
            f[mask] = pm_apply(pred[i][mask], tables[i])
            by_valid[anchor + timedelta(hours=i + 1)] = f
        out = {"anchor": anchor, "by_valid": by_valid, "mask": mask, "cycle": cycle}
        if keep_hrrr:  # verification wants the baseline; the live build does not
            out["hrrr_by_valid"] = {anchor + timedelta(hours=i + 1): h[i]
                                    for i in range(h.shape[0])}
        return out
    except Exception as e:  # noqa: BLE001 - the layer is optional, GFS is not
        print(f"radar model: skipped ({e})", file=sys.stderr)
        return None


if __name__ == "__main__":  # smoke test: python pipeline/radar_model.py
    import time
    from datetime import datetime, timezone

    import requests

    t = time.time()
    got = predict(requests.Session(), datetime.now(timezone.utc))
    if got is None:
        sys.exit("radar model layer unavailable")
    print(f"anchor {got['anchor']:%Y-%m-%d %H:%MZ}, HRRR cycle {got['cycle']:%HZ}, "
          f"{time.time()-t:.1f}s total")
    for valid, f in sorted(got["by_valid"].items()):
        wet = (f >= 23.0)[got["mask"]].mean()
        print(f"  {valid:%H:%M}  wet(>=23dBZ) {wet:.4f}  max {f.max():.1f} dBZ")
