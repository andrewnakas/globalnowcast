"""Hand-worked checks for verify/metrics.py. Run: python verify/test_metrics.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import bias, contingency, csi, far, fss, pod


def check(name, got, want, tol=1e-9):
    ok = (np.isnan(got) and np.isnan(want)) or abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r} want {want!r}")
    return ok


def main() -> int:
    ok = True

    # 4x4. Forecast and observation overlap in a known, countable way.
    pred = np.zeros((4, 4)); obs = np.zeros((4, 4))
    pred[0, 0:3] = 30.0          # 3 cells forecast wet
    obs[0, 1:4] = 30.0           # 3 cells observed wet, overlapping in 2
    mask = np.ones((4, 4), bool)

    h, m, f, cn = contingency(pred, obs, 20.0, mask)
    print("contingency")
    ok &= check("hits", h, 2)
    ok &= check("misses", m, 1)
    ok &= check("false alarms", f, 1)
    ok &= check("correct negatives", cn, 12)

    print("scores")
    ok &= check("csi", csi(h, m, f), 2 / 4)
    ok &= check("pod", pod(h, m), 2 / 3)
    ok &= check("far", far(h, m, f), 1 / 3)
    ok &= check("bias", bias(h, m, f), 3 / 3)

    # The mask must exclude cells entirely, not merely treat them as dry.
    print("masking")
    half = np.zeros((4, 4), bool); half[0, :] = True
    h2, m2, f2, cn2 = contingency(pred, obs, 20.0, half)
    ok &= check("masked hits", h2, 2)
    ok &= check("masked correct negatives", cn2, 0)

    # Degenerate cases must not raise.
    print("degenerate")
    dry = np.zeros((4, 4))
    ok &= check("csi with nothing anywhere", csi(*contingency(dry, dry, 20.0, mask)[:3]), float("nan"))

    print("fss")
    ok &= check("identical fields", fss(obs, obs, 20.0, 3, mask), 1.0)
    # Disjoint fields far apart share no neighbourhood -> no skill.
    a = np.zeros((9, 9)); b = np.zeros((9, 9))
    a[0, 0] = 30.0; b[8, 8] = 30.0
    ok &= check("disjoint fields", fss(a, b, 20.0, 3, np.ones((9, 9), bool)), 0.0)
    # A one-cell displacement should still score well at a tolerant window.
    c = np.zeros((9, 9)); d = np.zeros((9, 9))
    c[4, 4] = 30.0; d[4, 5] = 30.0
    near = fss(c, d, 20.0, 5, np.ones((9, 9), bool))
    ok &= check("1-cell shift beats 0.5 at window 5", near > 0.5, True)

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
