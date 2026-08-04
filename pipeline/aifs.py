"""ECMWF AIFS precipitation from the dynamical.org catalog, on the GFS grid.

Measured against RRQPE on the harvested pairs (ml/eval_aifs_vs_gfs.py), AIFS
beats GFS by +25/+36/+17% CSI at 5/10/20 dBZ and loses 41% at 30 dBZ: better at
ordinary rain, worse at heavy cores, which is the usual signature of a model
trained on a smooth objective. So this exists to be combined with GFS rather
than to replace it outright - see ml/eval_aifs_blend.py for which combination
earned its place.

AIFS ships a precipitation rate, not reflectivity, so it is converted through
the same Marshall-Palmer as every other observation path here; mixing dBZ
definitions across sources is the fastest way to get a blend subtly wrong.

Nothing here may raise into the hourly build: every entry point returns None on
failure, so an outage costs the AIFS arm and leaves GFS untouched.
"""
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import obs  # noqa: E402

BUCKET = "dynamical-ecmwf-aifs-single"
PREFIX = "ecmwf-aifs-single-forecast/v0.1.0.icechunk"
VAR = "precipitation_surface"
MAX_LEAD_H = 360
# AIFS publishes every 6 hours of lead, not hourly like GFS, so an hourly
# product has to interpolate between steps. Verified against the store: the
# grid is already row 0 = 90N and column 0 = -180.25, i.e. identical to the
# convention pipeline/render.decode_refc produces, so no reorientation is
# needed - but the check below stays, because a silent flip would be invisible
# in the blend and catastrophic in the output.
LEAD_STEP_H = 6
# The store is rebuilt per init; a cycle lands some hours after its init time,
# so the walk-back looks further than GFS's does.
CYCLE_HOURS = 6
SEARCH_CYCLES = 6

_ds = None
_lock = threading.Lock()


def _open():
    """Open (and memoise) the AIFS store, or None if unavailable/disabled."""
    global _ds
    if _ds is not None:
        return _ds or None
    with _lock:
        if _ds is not None:
            return _ds or None
        loaded = False
        if os.environ.get("NOWCAST_AIFS", "1") != "0":
            try:
                import icechunk
                import xarray as xr

                st = icechunk.s3_storage(bucket=BUCKET, prefix=PREFIX,
                                         region="us-west-2", anonymous=True)
                repo = icechunk.Repository.open(st)
                loaded = xr.open_zarr(repo.readonly_session("main").store,
                                      consolidated=False, chunks=None)
                print(f"aifs: opened {PREFIX}")
            except Exception as e:  # noqa: BLE001 - AIFS is optional
                print(f"aifs: disabled ({e})", file=sys.stderr)
        _ds = loaded
    return _ds or None


def is_active() -> bool:
    return _open() is not None


def _to_gfs_grid(rate, lat, lon):
    """Normalise orientation to row 0 = 90N, column 0 = -180, in rain rate."""
    if lat[0] < lat[-1]:
        rate, lat = rate[::-1], lat[::-1]
    if lon.max() > 180.0:
        roll = int((lon >= 180.0).argmax())
        rate = np.roll(rate, -roll, axis=1)
    return rate


def latest_cycle(now: datetime):
    """Newest init present in the store at or before `now`."""
    ds = _open()
    if ds is None:
        return None
    inits = ds.init_time.values
    cutoff = np.datetime64(now.replace(tzinfo=None), "ns")
    ok = inits[inits <= cutoff]
    return ok[-1] if len(ok) else None


def fetch(now: datetime, valids, retries: int = 3):
    """AIFS dBZ for each requested valid time, from the newest usable init.

    Returns ({valid: dbz on the 0.25 grid}, init) or None. Valid times outside
    the store's lead range are skipped rather than failing the batch.
    """
    ds = _open()
    if ds is None:
        return None
    try:
        init = latest_cycle(now)
        if init is None:
            return None
        init_dt = init.astype("datetime64[s]").astype(datetime).replace(
            tzinfo=timezone.utc)
        lat, lon = ds.latitude.values, ds.longitude.values

        # Read each needed 6-hourly step once, then interpolate: the store is
        # chunked per init, so re-reading a step per valid time would multiply
        # the network cost for no new data.
        steps = {}

        def step(lead_h):
            if lead_h in steps:
                return steps[lead_h]
            for attempt in range(retries):
                try:
                    sel = ds[VAR].sel(init_time=init,
                                      lead_time=np.timedelta64(lead_h, "h"))
                    rate = np.nan_to_num(sel.values.astype(np.float32)) * 3600.0
                    steps[lead_h] = _to_gfs_grid(rate, lat, lon)
                    return steps[lead_h]
                except Exception as e:  # noqa: BLE001 - flaky chunk reads
                    if attempt == retries - 1:
                        print(f"  aifs +{lead_h}h: {type(e).__name__}",
                              file=sys.stderr)
                        steps[lead_h] = None
                        return None
                    import time
                    time.sleep(3 * (attempt + 1))

        out = {}
        for valid in valids:
            lead = (valid - init_dt).total_seconds() / 3600.0
            if lead < 0 or lead > MAX_LEAD_H:
                continue
            lo = int(lead // LEAD_STEP_H) * LEAD_STEP_H
            hi = min(lo + LEAD_STEP_H, MAX_LEAD_H)
            a = step(lo)
            if a is None:
                continue
            if hi == lo or lead == lo:
                rate = a
            else:
                b = step(hi)
                # Interpolate in rain rate, never in dBZ - the same rule the
                # blend and the GFS interpolation follow.
                rate = a if b is None else \
                    (1 - (lead - lo) / (hi - lo)) * a + ((lead - lo) / (hi - lo)) * b
            out[valid] = obs.rain_to_dbz(rate)
        return (out, init_dt) if out else None
    except Exception as e:  # noqa: BLE001 - never break the build
        print(f"aifs: skipped ({e})", file=sys.stderr)
        return None


if __name__ == "__main__":  # smoke test: python pipeline/aifs.py
    import time

    now = datetime.now(timezone.utc)
    base = now.replace(minute=0, second=0, microsecond=0)
    want = [base + timedelta(hours=h) for h in (6, 12, 24, 48)]
    t = time.time()
    got = fetch(now, want)
    if got is None:
        sys.exit("aifs unavailable")
    fields, init = got
    print(f"init {init:%Y-%m-%d %HZ}, {len(fields)} frames, {time.time()-t:.1f}s")
    for valid, f in sorted(fields.items()):
        lead = int((valid - init).total_seconds() // 3600)
        print(f"  +{lead:3d}h  wet(>=20dBZ) {(f >= 20).mean():.4f}  "
              f"max {f.max():.1f} dBZ  shape {f.shape}")
