"""Categorical forecast verification: contingency tables and skill scores.

Pure numpy, no I/O, so it can be unit-tested against hand-worked examples.

Everything takes an explicit `mask`. That is not optional politeness: observations
only cover 521 of the 721 grid rows, and scoring the uncovered rows would count
empty polar cells as correct negatives and flatter every model equally.

Counts are returned raw so that many cases can be pooled into a single contingency
table before any ratio is taken. Averaging per-case CSI instead would be biased
toward cases with little rain.
"""
import numpy as np


def contingency(pred: np.ndarray, obs: np.ndarray, thr: float,
                mask: np.ndarray) -> tuple[int, int, int, int]:
    """(hits, misses, false alarms, correct negatives) above `thr`."""
    p = (pred >= thr) & mask
    o = (obs >= thr) & mask
    hits = int(np.count_nonzero(p & o))
    misses = int(np.count_nonzero(~p & o))
    fa = int(np.count_nonzero(p & ~o))
    cn = int(np.count_nonzero(mask)) - hits - misses - fa
    return hits, misses, fa, cn


def csi(h: int, m: int, f: int, *_) -> float:
    """Critical success index; hits over everything either side flagged."""
    d = h + m + f
    return h / d if d else float("nan")


def pod(h: int, m: int, *_) -> float:
    """Probability of detection."""
    d = h + m
    return h / d if d else float("nan")


def far(h: int, m: int, f: int, *_) -> float:
    """False alarm ratio."""
    d = h + f
    return f / d if d else float("nan")


def bias(h: int, m: int, f: int, *_) -> float:
    """Frequency bias: >1 means the forecast is too wet."""
    d = h + m
    return (h + f) / d if d else float("nan")


def _fractions(binary: np.ndarray, window: int) -> np.ndarray:
    """Mean of `binary` over a (window x window) box, via a summed-area table."""
    pad = window // 2
    a = np.pad(binary.astype(np.float64), pad, mode="constant")
    s = a.cumsum(axis=0).cumsum(axis=1)
    s = np.pad(s, ((1, 0), (1, 0)), mode="constant")
    h, w = binary.shape
    total = (s[window:window + h, window:window + w]
             - s[0:h, window:window + w]
             - s[window:window + h, 0:w]
             + s[0:h, 0:w])
    return total / (window * window)


def fss(pred: np.ndarray, obs: np.ndarray, thr: float, window: int,
        mask: np.ndarray) -> float:
    """Fractions skill score: agreement over a neighbourhood, not cell by cell.

    Tolerates small displacement errors, so it rewards forecasts that put the right
    weather slightly in the wrong place — which is exactly how advection fails.
    1.0 is perfect, 0.0 is no skill.
    """
    p = _fractions((pred >= thr) & mask, window)
    o = _fractions((obs >= thr) & mask, window)
    num = np.mean((p[mask] - o[mask]) ** 2)
    den = np.mean(p[mask] ** 2) + np.mean(o[mask] ** 2)
    if den == 0:
        return float("nan")
    return float(1.0 - num / den)


def scores(h: int, m: int, f: int, cn: int) -> dict:
    return {"hits": h, "misses": m, "false_alarms": f, "correct_neg": cn,
            "csi": csi(h, m, f), "pod": pod(h, m), "far": far(h, m, f),
            "bias": bias(h, m, f)}
