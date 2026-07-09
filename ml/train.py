"""Train the residual REFC-correction UNet on (GFS, obs) shards and export to ONNX.

Patch-based: random crops from each pair, so it trains fast and stays grid-agnostic.
Loss is masked to cells where observations exist (obs > FILL) so CONUS-only MRMS pairs
don't penalize the model for the empty ocean. Runs on one free GPU in hours.

  python train.py --shards shards/ --epochs 20 --out ../ml/model/refc_correction.onnx
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import RefcUNet  # noqa: E402

FILL = -30.0
PATCH = 128


class PatchPairs(Dataset):
    def __init__(self, shard_paths, patches_per_frame=8):
        self.gfs, self.obs = [], []
        for p in shard_paths:
            d = np.load(p)
            self.gfs.append(d["gfs"])
            self.obs.append(d["obs"])
        self.gfs = np.concatenate(self.gfs).astype(np.float32)
        self.obs = np.concatenate(self.obs).astype(np.float32)
        self.ppf = patches_per_frame
        self.H, self.W = self.gfs.shape[1:]

    def __len__(self):
        return len(self.gfs) * self.ppf

    def __getitem__(self, i):
        f = i // self.ppf
        y = np.random.randint(0, self.H - PATCH + 1)
        x = np.random.randint(0, self.W - PATCH + 1)
        g = self.gfs[f, y:y + PATCH, x:x + PATCH]
        o = self.obs[f, y:y + PATCH, x:x + PATCH]
        return torch.from_numpy(g[None]), torch.from_numpy(o[None])


def masked_l1(pred_dbz, target, mask):
    if mask.sum() == 0:
        return (pred_dbz * 0).sum()  # no obs in this patch -> zero, keeps grad graph
    return F.l1_loss(pred_dbz[mask], target[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "model" / "refc_correction.onnx"))
    args = ap.parse_args()

    paths = sorted(glob.glob(str(Path(args.shards) / "*.npz")))
    if not paths:
        sys.exit(f"no shards in {args.shards}")
    n_val = max(1, len(paths) // 5)
    train_ds, val_ds = PatchPairs(paths[n_val:] or paths), PatchPairs(paths[:n_val])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RefcUNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device={dev}  train_frames={len(train_ds)//train_ds.ppf}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for g, o in train_dl:
            g, o = g.to(dev), o.to(dev)
            pred = g + model(g)                 # residual correction
            loss = masked_l1(pred, o, o > FILL)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        vloss = evaluate(model, val_dl, dev)
        print(f"epoch {ep+1}/{args.epochs}  train {tot/len(train_dl):.3f}  val {vloss:.3f}")

    export_onnx(model, args.out)


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    tot, n = 0.0, 0
    for g, o in dl:
        g, o = g.to(dev), o.to(dev)
        pred = g + model(g)
        m = o > FILL
        if m.sum():
            tot += F.l1_loss(pred[m], o[m]).item()
            n += 1
    return tot / max(1, n)


def export_onnx(model, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.zeros(1, 1, 256, 256)
    kw = dict(
        input_names=["refc"], output_names=["residual"], opset_version=18,
        dynamic_axes={"refc": {2: "h", 3: "w"}, "residual": {2: "h", 3: "w"}},
    )
    # The legacy (dynamo=False) exporter honors dynamic_axes reliably, which we need so
    # one exported graph runs on both 128x128 patches and the full 721x1440 frame.
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(model, dummy, str(out), **kw)

    # Re-save with weights embedded so the model is a single portable .onnx file
    # (no .onnx.data sidecar to ship), and drop any sidecar the exporter emitted.
    import onnx

    m = onnx.load(str(out))
    onnx.save_model(m, str(out), save_as_external_data=False)
    sidecar = out.with_suffix(out.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()

    # Verify the export accepts a non-training spatial size before we trust it.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    probe = np.zeros((1, 1, 720, 1440), dtype=np.float32)
    sess.run(None, {sess.get_inputs()[0].name: probe})
    print(f"exported ONNX -> {out} ({out.stat().st_size/1e6:.2f} MB), dynamic shapes verified")


if __name__ == "__main__":
    main()
