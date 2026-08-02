"""Is ECMWF AIFS a better model arm for the blend than GFS?

Every measurement in this project says GFS is the weak link: CSI ~0.19 flat
across leads, a 1.7-1.9x wet-area bias, and worst of all in the tropics. AIFS is
ECMWF's AI forecast, 0.25 degrees, 0-360 h, 6-hourly inits, carrying
precipitation_surface - so it can be scored on exactly the cases already
harvested for the correction model (ml/gfs_pairs holds GFS REFC plus the RRQPE
observation at each valid time).

The comparison has to be like for like. GFS ships as composite reflectivity;
AIFS gives a precipitation rate. Both are converted to dBZ through the same
Marshall-Palmer used everywhere else, and the RRQPE observation is the same
truth the shipped blend is scored against. Scored only where the satellite sees
(the pairs' own mask), pooled over every lead.

    python ml/eval_aifs_vs_gfs.py

If AIFS wins at the render thresholds it becomes a candidate model arm; the
blend would then need its handover refitted, because a stronger model arm moves
the crossover earlier.
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE.parent / "verify"))

from metrics import contingency, csi  # noqa: E402

U8_LO, U8_HI = -30.0, 60.0
THRESHOLDS = (5.0, 10.0, 20.0, 30.0)
BUCKET = "dynamical-ecmwf-aifs-single"
PREFIX = "ecmwf-aifs-single-forecast/v0.1.0.icechunk"
VAR = "precipitation_surface"


def from_u8(a):
    return a.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO


def rain_to_dbz(rate):
    """Marshall-Palmer, identical to pipeline/obs.py so every path shares a dBZ."""
    out = np.full(rate.shape, U8_LO, np.float32)
    wet = rate > 0.01
    out[wet] = 10.0 * np.log10(200.0 * np.power(rate[wet], 1.6))
    return np.maximum(out, U8_LO)


def open_aifs():
    import icechunk
    import xarray as xr

    st = icechunk.s3_storage(bucket=BUCKET, prefix=PREFIX, region="us-west-2",
                             anonymous=True)
    repo = icechunk.Repository.open(st)
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False,
                        chunks=None)


def aifs_dbz(ds, init, lead_h, retries=4):
    """AIFS precipitation at (init, lead) as dBZ on the GFS grid convention.

    The store is latitude-descending or ascending depending on the build, and
    longitude may run 0..360; both are normalised to row 0 = 90N, column 0 =
    -180 so the field lines up with the harvested GFS and RRQPE arrays.
    """
    for attempt in range(retries):
        try:
            sel = ds[VAR].sel(init_time=np.datetime64(init, "ns"),
                              lead_time=np.timedelta64(int(lead_h), "h"))
            rate = np.nan_to_num(sel.values.astype(np.float32)) * 3600.0
            break
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  aifs {init} +{lead_h}h: {type(e).__name__}",
                      file=sys.stderr)
                return None
            import time
            time.sleep(4 * (attempt + 1))

    lat, lon = ds.latitude.values, ds.longitude.values
    if lat[0] < lat[-1]:
        rate, lat = rate[::-1], lat[::-1]
    if lon.max() > 180.0:
        roll = int((lon >= 180.0).argmax())
        rate = np.roll(rate, -roll, axis=1)
    return rain_to_dbz(rate)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="ml/gfs_pairs")
    ap.add_argument("--limit", type=int, default=12, help="valid times to score")
    args = ap.parse_args()

    ds = open_aifs()
    inits = ds.init_time.values
    print(f"aifs inits {str(inits[0])[:10]} .. {str(inits[-1])[:16]}, "
          f"{len(inits)} of them")

    paths = sorted(Path(args.pairs).glob("*.npz"))[-args.limit:]
    counts = {n: {t: np.zeros(3) for t in THRESHOLDS} for n in ("gfs", "aifs")}
    wet = {"gfs": 0.0, "aifs": 0.0, "obs": 0.0}
    used = 0

    for p in paths:
        valid = datetime.strptime(p.stem, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        z = np.load(p)
        obs_dbz, mask = from_u8(z["obs"]), z["mask"]
        scored_any = False
        for k in z.files:
            if not k.startswith("gfs_"):
                continue
            lead = int(k[4:])
            init = np.datetime64(valid.replace(tzinfo=None), "ns") \
                - np.timedelta64(lead, "h")
            if init not in inits:
                continue
            a = aifs_dbz(ds, init.astype("datetime64[ns]").item(), lead)
            if a is None or a.shape != obs_dbz.shape:
                continue
            g = from_u8(z[k])
            for name, f in (("gfs", g), ("aifs", a)):
                for t in THRESHOLDS:
                    counts[name][t] += contingency(f, obs_dbz, t, mask)[:3]
                wet[name] += float((f >= 20.0)[mask].sum())
            wet["obs"] += float((obs_dbz >= 20.0)[mask].sum())
            scored_any = True
        used += scored_any
        print(f"{p.stem}: {'scored' if scored_any else 'no matching inits'}",
              flush=True)

    if not used:
        sys.exit("no valid times scored - check the init/lead alignment")

    print(f"\n{used} valid times, pooled over leads 6-48h, truth = RRQPE")
    print(f"{'':>7}" + "".join(f"{t:>9g}" for t in THRESHOLDS) + f"{'wet vs obs':>12}")
    for n in ("gfs", "aifs"):
        print(f"{n:>7}" + "".join(f"{csi(*counts[n][t]):>9.4f}" for t in THRESHOLDS)
              + f"{wet[n] / max(wet['obs'], 1.0):>12.2f}")
    gains = [csi(*counts['aifs'][t]) / max(csi(*counts['gfs'][t]), 1e-9) - 1
             for t in THRESHOLDS]
    print("\naifs vs gfs: " + "  ".join(f"{t:g}dBZ {g:+.1%}"
                                        for t, g in zip(THRESHOLDS, gains)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
