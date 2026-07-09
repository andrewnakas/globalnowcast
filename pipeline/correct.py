"""Optional ML correction of GFS REFC toward observed precipitation.

Loads a small residual UNet exported to ONNX and applies it on CPU. This is a
no-op passthrough unless a trained model exists at ml/model/refc_correction.onnx
and onnxruntime is importable, so the pipeline runs unchanged before any model is
trained. Set NOWCAST_CORRECT=0 to force it off.
"""
import os
import threading
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "model" / "refc_correction.onnx"
DBZ_MIN, DBZ_MAX = -30.0, 80.0
_session = None
_enabled = None
_lock = threading.Lock()  # frames are rendered on a thread pool; build the session once


def _load():
    """Lazily build the ONNX session once (thread-safe); None if unavailable/disabled."""
    global _session, _enabled
    if _enabled is not None:
        return _session
    with _lock:
        if _enabled is not None:  # another thread finished while we waited
            return _session
        enabled = False
        session = None
        if os.environ.get("NOWCAST_CORRECT", "1") != "0" and MODEL_PATH.exists():
            try:
                import onnxruntime as ort

                # One intra-op thread per session: the frame pool already saturates the
                # CPU, so multi-threaded ORT would oversubscribe and thrash.
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                session = ort.InferenceSession(
                    str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
                )
                enabled = True
                print(f"correction: using {MODEL_PATH.name}")
            except Exception as e:  # noqa: BLE001 - never let correction break the run
                print(f"correction: disabled ({e})")
                session = None
        _session, _enabled = session, enabled
    return _session


def is_active() -> bool:
    return _load() is not None


def correct(dbz: np.ndarray) -> np.ndarray:
    """Apply the residual correction to a dBZ field, or return it unchanged.

    The model predicts a residual added to the input; missing data (large negative
    fill) is preserved. Input/output are the same 2-D dBZ array.
    """
    sess = _load()
    if sess is None:
        return dbz

    valid = dbz > DBZ_MIN
    h, w = dbz.shape
    # UNet pools 2x (see ml/model.py POOL_DIVISOR), so H and W must be multiples of 4;
    # pad (reflect) up to that, run, then crop back.
    ph, pw = (-h) % 4, (-w) % 4
    x = np.where(valid, dbz, 0.0).astype(np.float32)
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw)), mode="reflect")
    x = x[None, None]  # (1,1,H,W)
    try:
        residual = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    except Exception as e:  # noqa: BLE001
        print(f"correction: inference failed, passing through ({e})")
        return dbz
    residual = residual[:h, :w]
    out = np.where(valid, np.clip(dbz + residual, DBZ_MIN, DBZ_MAX), dbz)
    return out.astype(dbz.dtype)
