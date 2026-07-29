"""ML correction of GFS REFC toward observed precipitation, for leads beyond 6 h.

v2, trained on live-harvested pairs (ml/data/build_gfs_pairs.py + ml/train_correction.py):
input channels are GFS dBZ, lead/48 and |lat|/90, and the output field is
probability-matched to the observed dBZ climatology so the L1-hedged residual does
not delete heavy cores (measured: without PM, 30 dBZ CSI collapses 0.079 -> 0.008;
with it, 0.102 - above raw at every render threshold).

Two hard rules, both measured:
  * Leads <= MIN_LEAD_H are returned untouched. The 0-4.5h satellite blend must see
    raw GFS - frequency-matching GFS before blending lost in 45/45 cases.
  * PM applies only inside the satellite-observed rows (70N-60S); poleward the
    corrected field passes through un-matched, since the reference climatology
    knows nothing about polar precipitation.

No-op passthrough unless ml/model/refc_correction_v2.onnx AND correction_pm.npz
exist and onnxruntime imports. Set NOWCAST_CORRECT=0 to force it off.
"""
import os
import threading
from pathlib import Path

import numpy as np

from obs import GLOBAL

_MODEL_DIR = Path(__file__).resolve().parent.parent / "ml" / "model"
MODEL_PATH = _MODEL_DIR / "refc_correction_v2.onnx"
PM_PATH = _MODEL_DIR / "correction_pm.npz"
DBZ_MIN, DBZ_MAX = -30.0, 80.0
LAT_CH = np.abs(np.linspace(90.0, -90.0, 721, dtype=np.float32))[:, None] / 90.0

_state = None  # None = untried; (session, table) = ready; False = disabled
_lock = threading.Lock()


def _load():
    global _state
    if _state is not None:
        return _state or None
    with _lock:
        if _state is not None:
            return _state or None
        loaded = False
        if (os.environ.get("NOWCAST_CORRECT", "1") != "0"
                and MODEL_PATH.exists() and PM_PATH.exists()):
            try:
                import onnxruntime as ort

                # One intra-op thread: the frame pool already saturates the CPU.
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                session = ort.InferenceSession(str(MODEL_PATH), sess_options=opts,
                                               providers=["CPUExecutionProvider"])
                table = np.load(PM_PATH)["ref_q"].astype(np.float32)
                loaded = (session, table)
                print(f"correction: using {MODEL_PATH.name} + {PM_PATH.name}")
            except Exception as e:  # noqa: BLE001 - never break the run
                print(f"correction: disabled ({e})")
        _state = loaded
    return _state or None


def is_active() -> bool:
    return _load() is not None


def pm_apply(pred: np.ndarray, ref_q: np.ndarray) -> np.ndarray:
    """Rank-preserving distribution replacement; see ml/export_pm.py."""
    flat = pred.reshape(-1)
    order = np.argsort(np.argsort(flat, kind="stable"), kind="stable")
    idx = np.linspace(0, ref_q.size - 1, flat.size).astype(np.int64)
    ref = ref_q[np.clip(idx, 0, ref_q.size - 1)]
    return ref[order].reshape(pred.shape)


def correct(dbz: np.ndarray, lead_h: int = 0, blend_frame: bool = False) -> np.ndarray:
    """Correct one GFS REFC field, or return it unchanged.

    `lead_h` is the GFS forecast hour (the model's lead channel). `blend_frame`
    marks frames the satellite blend consumes - those pass through raw, because
    the gate is which *product window* the frame feeds, not its nominal lead: an
    aged cycle gives the blend frames at GFS leads well past 6 h.
    """
    got = _load()
    if got is None or blend_frame:
        return dbz
    sess, table = got

    h, w = dbz.shape
    valid = dbz > -900.0  # decode_refc fills gaps with -999
    g = np.where(valid, np.maximum(dbz, DBZ_MIN), DBZ_MIN).astype(np.float32)
    lat = np.broadcast_to(LAT_CH[:h], (h, w)).astype(np.float32)
    # The lead channel saw 6-48 h in training; an aged cycle's far tail can sit a
    # few hours past that, where clipping beats extrapolating.
    x = np.stack([g, np.full((h, w), min(lead_h, 48) / 48.0, np.float32), lat])
    ph, pw = (-h) % 4, (-w) % 4  # two pooling levels (ml/model.POOL_DIVISOR)
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    try:
        res = sess.run(None, {sess.get_inputs()[0].name: x[None]})[0][0, 0]
    except Exception as e:  # noqa: BLE001
        print(f"correction: inference failed, passing through ({e})")
        return dbz
    out = np.clip(g + res[:h, :w], DBZ_MIN, DBZ_MAX)

    # Restore the intensity distribution where the reference climatology applies.
    obs_rows = GLOBAL.observed_rows[:h]
    region = obs_rows[:, None] & valid
    out[region] = pm_apply(out[region], table)
    return np.where(valid, out, dbz).astype(dbz.dtype)
