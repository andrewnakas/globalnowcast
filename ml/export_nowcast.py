"""Export a trained nowcast checkpoint to ONNX for the hourly job.

The site build runs on GitHub Actions with onnxruntime and no torch - requirements.txt
deliberately keeps torch out, since it is a ~800 MB install for a job that otherwise
takes seconds. So a .pt checkpoint cannot ship as-is; it has to become a single
self-contained .onnx file.

    python ml/export_nowcast.py --ckpt ml/model/nowcast_all32_s1.pt

Verifies the export rather than trusting it: loads the result in onnxruntime, runs it
against the torch model on the same input, and fails if they disagree. A silent
export mismatch would show up as a quietly worse forecast rather than an error.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nowcast_model import IN_FRAMES, OUT_FRAMES, NowcastUNet  # noqa: E402

TOL = 1e-3  # dBZ; the render bins are 5 dBZ wide, so this is far below one bin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ml/model/nowcast_all32_s1.pt")
    ap.add_argument("--out", default=None, help="defaults to <ckpt>.onnx")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    out = Path(args.out or str(Path(args.ckpt).with_suffix(".onnx")))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = NowcastUNet(base=ck.get("base", 32), gated=ck.get("gated", False))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"{args.ckpt}: base={ck.get('base')} gated={ck.get('gated')} "
          f"val_csi={ck.get('val_csi', float('nan')):.4f}")

    # Dynamic height and width: the same graph has to serve 100x100 training tiles,
    # a 521x1440 global grid and a 2 km regional window.
    #
    # The dummy shape must NOT be a multiple of 8. Tracing only records the branches
    # it executes, so exporting at 128x128 - where (-128) % 8 == 0 - skips the
    # reflect-pad entirely and produces a graph that silently fails on every shape
    # that needs padding. 100x100 pads by 4, so the padding is traced in.
    radar = torch.zeros(1, IN_FRAMES, 100, 100)
    hrrr = torch.zeros(1, OUT_FRAMES, 100, 100)
    dyn = {2: "h", 3: "w"}
    kw = dict(input_names=["radar", "hrrr"], output_names=["forecast"],
              dynamic_axes={"radar": dyn, "hrrr": dyn, "forecast": dyn},
              opset_version=args.opset)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(model, (radar, hrrr), str(out), dynamo=False, **kw)
    except TypeError:  # older torch has no dynamo argument
        torch.onnx.export(model, (radar, hrrr), str(out), **kw)

    # Embed the weights so a single file ships, and drop any sidecar the exporter left.
    import onnx
    onnx.save_model(onnx.load(str(out)), str(out), save_as_external_data=False)
    sidecar = out.with_suffix(out.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()

    # Verify against torch at a shape the exporter never saw, so the dynamic axes are
    # actually exercised rather than assumed.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for h, w in ((100, 100), (144, 208)):
        r = rng.uniform(-30, 60, (1, IN_FRAMES, h, w)).astype(np.float32)
        f = rng.uniform(-30, 60, (1, OUT_FRAMES, h, w)).astype(np.float32)
        got = sess.run(["forecast"], {"radar": r, "hrrr": f})[0]
        with torch.no_grad():
            want = model(torch.from_numpy(r), torch.from_numpy(f)).numpy()
        err = float(np.abs(got - want).max())
        status = "ok" if err < TOL else "MISMATCH"
        print(f"  {h}x{w}: max |onnx - torch| = {err:.2e}  {status}")
        if err >= TOL:
            return 1

    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("The hourly job needs onnxruntime only; torch stays out of requirements.txt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
