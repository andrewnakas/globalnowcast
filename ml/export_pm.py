"""Export the probability-matching tables to .npz for the hourly job.

Probability matching is a shipping requirement, not a refinement: the raw model
over-paints light rain 1.53x and produces only 41% of the heavy rain present, and PM
fixes both by replacing the predicted intensity distribution with the reference
climatology per lead. The tables were fitted with torch (ml/model/pm_tables.pt), but
the hourly job runs with onnxruntime and numpy only, so - like the model itself - the
tables have to ship in a torch-free format.

    python ml/export_pm.py --ckpt ml/model/nowcast_all32_s1.pt

Verifies the export rather than trusting it: runs the checkpoint over held-out tiles
and scores torch probability_match against the numpy port on the same predictions.
The two differ only in how ties are ordered, which must not move CSI.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "verify"))

THRESHOLDS = {"1 mm/h": 23.0, "4 mm/h": 32.6, "8 mm/h": 37.5}
TOL = 0.003  # replicate noise at 48k data; a faithful port sits far below this


def pm_apply(pred: np.ndarray, ref_q: np.ndarray) -> np.ndarray:
    """Torch-free port of nowcast_model.probability_match for one lead.

    Same construction: each cell keeps its rank within the prediction and takes the
    value at that rank from the reference distribution, resampled to the cell count.
    Apply to the full field for a lead, never tile-by-tile: matching a small dry
    tile to the climatology would invent rain in it.
    """
    flat = pred.reshape(-1)
    order = np.argsort(np.argsort(flat, kind="stable"), kind="stable")
    idx = np.linspace(0, ref_q.size - 1, flat.size).astype(np.int64)
    ref = ref_q[np.clip(idx, 0, ref_q.size - 1)]
    return ref[order].reshape(pred.shape)


def main() -> int:
    # Deferred: pm_apply above is pure numpy and gets imported by torch-free
    # serving/eval code; only the export+verify path needs torch.
    import torch

    from metrics import contingency, csi
    from nowcast_model import NowcastUNet, probability_match
    from train_nowcast import Tiles, all_months, from_u8

    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="ml/model/pm_tables.pt")
    ap.add_argument("--ckpt", default="ml/model/nowcast_all32_s1.pt")
    ap.add_argument("--tiles", default="ml/tiles_all")
    ap.add_argument("--val-months", type=int, default=12)
    ap.add_argument("--limit", type=int, default=1500,
                    help="held-out samples to verify on (CPU-friendly)")
    ap.add_argument("--out", default="ml/model/pm_tables.npz")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch CPU threads; keep low so a running trainer is unhurt")
    args = ap.parse_args()

    d = torch.load(args.tables, map_location="cpu", weights_only=False)
    tables = d["tables"].numpy().astype(np.float32)  # (OUT_FRAMES, n) sorted dBZ
    np.savez(args.out, ref_q=tables, n=d.get("n", tables.shape[1]),
             fitted_on=d.get("fitted_on", -1), source=str(args.tables))
    print(f"wrote {args.out}: ref_q {tables.shape}, "
          f"fitted on {d.get('fitted_on', '?')} samples")

    # Verify on held-out predictions: torch PM vs the numpy port must agree.
    torch.set_num_threads(args.threads)
    paths = sorted(Path(args.tiles).glob("*.npz"))
    months = all_months(paths)
    step = max(1, len(months) // max(args.val_months, 1))
    vm = set(months[::step][:args.val_months])
    val = Tiles(paths, vm, exclude=False)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = NowcastUNet(base=ck.get("base", 32), gated=ck.get("gated", False))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    n = min(args.limit, len(val.x))
    x = torch.from_numpy(from_u8(val.x[:n]))
    h = torch.from_numpy(from_u8(val.h[:n]))
    preds = []
    with torch.no_grad():
        for i in range(0, n, 32):
            preds.append(model(x[i:i + 32], h[i:i + 32]))
    pred = torch.cat(preds)
    truth = from_u8(val.y[:n])

    tab = torch.from_numpy(tables)
    print(f"\n{n} held-out samples")
    worst = 0.0
    for label, thr in THRESHOLDS.items():
        print(f"\nCSI at {label}")
        print(f"{'lead':>6}{'torch PM':>10}{'numpy PM':>10}{'diff':>9}")
        for k in range(truth.shape[1]):
            t = truth[:, k]
            m = np.ones_like(t, bool)
            a = probability_match(pred[:, k], tab[k]).numpy()
            b = pm_apply(pred[:, k].numpy(), tables[k])
            ca = csi(*contingency(a, t, thr, m)[:3])
            cb = csi(*contingency(b, t, thr, m)[:3])
            worst = max(worst, abs(ca - cb))
            print(f"{'+' + str(k + 1) + 'h':>6}{ca:>10.4f}{cb:>10.4f}{ca - cb:>9.4f}")

    print(f"\nmax |torch - numpy| CSI = {worst:.4f}  "
          f"({'ok' if worst < TOL else 'MISMATCH'}, tol {TOL})")
    return 0 if worst < TOL else 1


if __name__ == "__main__":
    sys.exit(main())
