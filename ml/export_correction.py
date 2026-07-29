"""Export the GFS correction model + its PM table for the hourly job.

Writes ml/model/refc_correction_v2.onnx (3-channel input: dBZ, lead/48, |lat|/90,
dynamic H/W) and ml/model/correction_pm.npz (the observed-dBZ climatology quantiles
fitted on the training pairs). Verifies the ONNX against torch before writing, in
the house style: a silent export mismatch is a quietly worse forecast.

    python ml/export_correction.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import RefcUNet  # noqa: E402

# Accumulation differences grow with spatial extent (onnxruntime fuses convs
# differently than eager torch); at the full 721x1440 frame the max divergence is
# ~6e-3 dBZ. The render bins are 5 dBZ wide, so anything under ~0.1 dBZ is
# invisible; the tolerance is set well below that, not at float epsilon.
TOL = 5e-2
U8_LO, U8_HI = -30.0, 60.0


def main() -> int:
    ck = torch.load("ml/model/refc_correction_v2.pt", map_location="cpu",
                    weights_only=False)
    net = RefcUNet(base=ck["base"], cin=ck["cin"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    print(f"checkpoint: epoch {ck['epoch']}, mean CSI gain {ck['mean_gain']:+.4f}")

    out = Path("ml/model/refc_correction_v2.onnx")
    x = torch.zeros(1, 3, 100, 100)  # not a multiple of 4: pad path gets traced
    dyn = {0: "batch", 2: "h", 3: "w"}
    kw = dict(input_names=["x"], output_names=["residual"],
              dynamic_axes={"x": dyn, "residual": dyn}, opset_version=18)
    try:
        torch.onnx.export(net, (x,), str(out), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(net, (x,), str(out), **kw)
    import onnx
    onnx.save_model(onnx.load(str(out)), str(out), save_as_external_data=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for b, h, w in ((1, 100, 100), (2, 96, 152), (1, 724, 1440)):
        a = rng.uniform(-30, 60, (b, 3, h, w)).astype(np.float32)
        got = sess.run(["residual"], {"x": a})[0]
        with torch.no_grad():
            want = net(torch.from_numpy(a)).numpy()
        err = float(np.abs(got - want).max())
        print(f"  b{b} {h}x{w}: max |onnx - torch| = {err:.2e}"
              f"  {'ok' if err < TOL else 'MISMATCH'}")
        if err >= TOL:
            return 1

    # PM reference: observed climatology from the training pairs only.
    f8 = lambda arr: arr.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO
    paths = sorted(Path("ml/gfs_pairs").glob("*.npz"))[:-10]
    qs = []
    for p in paths[::4]:
        z = np.load(p)
        qs.append(np.quantile(f8(z["obs"])[z["mask"]], np.linspace(0, 1, 2048)))
    table = np.mean(qs, axis=0).astype(np.float32)
    np.savez("ml/model/correction_pm.npz", ref_q=table,
             fitted_on=len(paths[::4]), source="ml/gfs_pairs train obs")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) + correction_pm.npz "
          f"(wet>=20dBZ {(table >= 20).mean():.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
