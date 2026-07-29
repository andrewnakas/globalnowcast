"""Does probability matching restore the correction's heavy-rain tail?

The trained correction lifts CSI at 5/10/20 dBZ substantially but collapses
30 dBZ (0.079 -> 0.008): L1 hedging deletes heavy cores, the same failure PM
fixed for the CONUS radar model. Here the reference is the observed dBZ
climatology from the *training* pairs, applied to the corrected field over the
observation mask - the held-out times never touch the fit.

    python ml/eval_correction_pm.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "verify"))

from export_pm import pm_apply  # noqa: E402
from metrics import contingency, csi  # noqa: E402
from model import POOL_DIVISOR, RefcUNet  # noqa: E402

U8_LO, U8_HI = -30.0, 60.0
THRESHOLDS = (5.0, 10.0, 20.0, 30.0)
LAT = np.linspace(90.0, -90.0, 721, dtype=np.float32)


def f8(a):
    return a.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO


def main() -> int:
    paths = sorted(Path("ml/gfs_pairs").glob("*.npz"))
    train_p, val_p = paths[:-10], paths[-10:]

    qs = []
    for p in train_p[::4]:
        z = np.load(p)
        qs.append(np.quantile(f8(z["obs"])[z["mask"]], np.linspace(0, 1, 2048)))
    table = np.mean(qs, axis=0).astype(np.float32)
    print(f"obs table wet frac >=20dBZ: {(table >= 20).mean():.4f}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("ml/model/refc_correction_v2.pt", map_location=dev,
                    weights_only=False)
    net = RefcUNet(base=ck["base"], cin=ck["cin"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    counts = {n: {t: np.zeros(3) for t in THRESHOLDS}
              for n in ("raw", "corr", "corr_pm")}
    with torch.no_grad():
        for p in val_p:
            z = np.load(p)
            o, m = f8(z["obs"]), z["mask"]
            for k in z.files:
                if not k.startswith("gfs_"):
                    continue
                g = f8(z[k])
                lead = float(k[4:])
                lat = np.broadcast_to(np.abs(LAT[:, None]) / 90.0,
                                      g.shape).astype(np.float32)
                x = np.stack([g, np.full_like(g, lead / 48.0), lat])
                ph, pw = (-g.shape[0]) % POOL_DIVISOR, (-g.shape[1]) % POOL_DIVISOR
                xp = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
                res = net(torch.from_numpy(xp[None]).to(dev))
                res = res[0, 0, :g.shape[0], :g.shape[1]].cpu().numpy()
                c = np.clip(g + res, U8_LO, 80.0)
                cp = c.copy()
                cp[m] = pm_apply(c[m], table)
                for n, f in (("raw", g), ("corr", c), ("corr_pm", cp)):
                    for t in THRESHOLDS:
                        counts[n][t] += contingency(f, o, t, m)[:3]

    print("\npooled held-out CSI")
    print(" " * 9 + "".join(f"{t:>9g}" for t in THRESHOLDS))
    for n in ("raw", "corr", "corr_pm"):
        print(f"{n:>9}" + "".join(f"{csi(*counts[n][t]):>9.4f}"
                                  for t in THRESHOLDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
