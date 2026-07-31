"""Cross-truth the GFS correction against IMERG Late - a different instrument.

The correction was trained and gated against the RRQPE geostationary retrieval;
a gain that only exists against its own training truth could be retrieval
agreement rather than skill. IMERG Late merges passive microwave, IR and *gauge
adjustment* - independent errors - so a gain that survives here is real. This is
the multi-sensor check the plan flagged to avoid single-truth circularity.

Runs on the box with icechunk (the 1080):

    python ml/eval_correction_imerg.py
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
BUCKET, PREFIXES = "dynamical-nasa-imerg", (
    "nasa-imerg-analysis-late/v0.1.0.icechunk",
    "nasa-imerg-late/v0.1.0.icechunk",
    "nasa-imerg-analysis-late/v0.2.0.icechunk",
)


def f8(a):
    return a.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO


def rain_to_dbz(rate):
    out = np.full(rate.shape, U8_LO, np.float32)
    wet = rate > 0.01
    out[wet] = 10.0 * np.log10(200.0 * np.power(rate[wet], 1.6))
    return np.maximum(out, U8_LO)


def open_imerg():
    """Open the IMERG Late store, with timeouts sized for its chunks.

    icechunk's default 3.1 s connect timeout loses this store repeatedly: the
    chunks are global 0.1-degree fields and S3 is slow to first byte under any
    concurrent load. Raising the timeouts turns a hard failure into a wait.
    """
    import icechunk
    import xarray as xr

    cfg = icechunk.RepositoryConfig.default()
    try:
        from datetime import timedelta as _td

        cfg.storage = icechunk.StorageSettings(
            retries=icechunk.StorageRetriesSettings(max_tries=8),
            timeouts=icechunk.StorageTimeoutSettings(
                connect_timeout=_td(seconds=30),
                read_timeout=_td(seconds=180)))
    except Exception as e:  # noqa: BLE001 - fall back to library defaults
        print(f"  (storage settings unavailable: {type(e).__name__})",
              file=sys.stderr)
        cfg = None

    for pfx in PREFIXES:
        try:
            st = icechunk.s3_storage(bucket=BUCKET, prefix=pfx,
                                     region="us-west-2", anonymous=True)
            repo = (icechunk.Repository.open(st, config=cfg) if cfg
                    else icechunk.Repository.open(st))
            ds = xr.open_zarr(repo.readonly_session("main").store,
                              consolidated=False, chunks=None)
            print(f"imerg store: {pfx}")
            return ds
        except Exception as e:  # noqa: BLE001
            print(f"  {pfx}: {type(e).__name__}", file=sys.stderr)
    raise SystemExit("no IMERG store found; check the prefix against the STAC")


def imerg_dbz(ds, when):
    """IMERG Late at `when` as MP dBZ on the 0.25-degree GFS grid."""
    from scipy.ndimage import zoom

    sel = ds["precipitation_surface"].sel(time=np.datetime64(when, "ns"),
                                          method="nearest")
    if abs((sel.time.values - np.datetime64(when, "ns"))
           / np.timedelta64(1, "m")) > 20:
        return None
    # One read still fails often enough to lose a whole run; retry rather than
    # abandon the valid time.
    for attempt in range(4):
        try:
            rate = np.nan_to_num(sel.values.astype(np.float32)) * 3600.0
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"  {when}: {type(e).__name__} after 4 tries",
                      file=sys.stderr)
                return None
            import time
            time.sleep(5 * (attempt + 1))
    # Standardise to the GFS convention (row 0 = 90N), then downsample 0.1 ->
    # 0.25 in rate space: a 2x2 block mean first (exact), then a small bilinear
    # zoom for the remaining 1.25x - averaging before interpolating keeps the
    # cell-mean semantics of the coarse grid.
    if ds.latitude.values[0] < ds.latitude.values[-1]:
        rate = rate[::-1]
    rate = rate[:1800:, :].reshape(900, 2, 1800, 2).mean(axis=(1, 3))
    out = zoom(rate, (721 / 900, 1440 / 1800), order=1)
    return rain_to_dbz(np.maximum(out, 0.0))


def main() -> int:
    ds = open_imerg()
    paths = sorted(Path("ml/gfs_pairs").glob("*.npz"))[-10:]  # held-out times

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("ml/model/refc_correction_v2.pt", map_location=dev,
                    weights_only=False)
    net = RefcUNet(base=ck["base"], cin=ck["cin"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    table = np.load("ml/model/correction_pm.npz")["ref_q"].astype(np.float32)

    counts = {n: {t: np.zeros(3) for t in THRESHOLDS}
              for n in ("raw", "corr_pm")}
    used = 0
    from datetime import datetime, timezone
    with torch.no_grad():
        for p in paths:
            when = datetime.strptime(p.stem, "%Y%m%d%H").replace(
                tzinfo=timezone.utc)
            truth = imerg_dbz(ds, when.replace(tzinfo=None))
            if truth is None:
                continue
            z = np.load(p)
            m = z["mask"]  # score on the RRQPE-observed rows for comparability
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
                c[m] = pm_apply(c[m], table)
                for n, f in (("raw", g), ("corr_pm", c)):
                    for t in THRESHOLDS:
                        counts[n][t] += contingency(f, truth, t, m)[:3]
            used += 1
            print(f"{p.stem}: scored", flush=True)

    print(f"\n{used} held-out valid times vs IMERG Late "
          f"(microwave+IR+gauge - independent of the training truth)")
    print(" " * 9 + "".join(f"{t:>9g}" for t in THRESHOLDS))
    for n in ("raw", "corr_pm"):
        print(f"{n:>9}" + "".join(f"{csi(*counts[n][t]):>9.4f}"
                                  for t in THRESHOLDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
