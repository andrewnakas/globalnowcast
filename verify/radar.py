"""MRMS radar truth, on whatever grid we are scoring.

Verifying satellite forecasts against later satellite observations measures internal
consistency: both sides share the retrieval's own errors, so a bias in the retrieval
is invisible. Published benchmarks all verify against radar, which is an independent
instrument, and that is the comparison that actually answers "how do we stack up".

MRMS MergedReflectivityQCComposite is CONUS-only, ~1 km, every ~2 min, anonymous S3,
native dBZ - so no Z-R conversion is needed on this side and the units already match
what the pipeline produces.

Fetching is deliberately not shared with ml/data/mrms.py: that one regrids onto the
fixed GFS grid for training, while this has to land on an arbitrary verification grid.
"""
import gzip
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import obs  # noqa: E402

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCT = "MergedReflectivityQCComposite_00.50"
FILL = obs.FILL
MAX_AGE_S = 300  # a "truth" frame more than 5 min off the target time is not truth


def closest_key(session: requests.Session, when: datetime) -> tuple[str, datetime]:
    """The MRMS object nearest `when`, with its actual timestamp."""
    prefix = f"CONUS/{PRODUCT}/{when:%Y%m%d}/"
    r = session.get(f"{BUCKET}?list-type=2&prefix={prefix}&max-keys=1000", timeout=60)
    r.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [e.text for e in ET.fromstring(r.content).findall(".//s3:Key", ns)]
    keys = [k for k in keys if k.endswith(".grib2.gz")]
    if not keys:
        raise RuntimeError(f"no MRMS objects under {prefix}")

    def stamp(k: str) -> datetime:
        t = k.rsplit("_", 1)[-1].replace(".grib2.gz", "")  # YYYYMMDD-HHMMSS
        return datetime.strptime(t, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)

    ref = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    best = min(keys, key=lambda k: abs((stamp(k) - ref).total_seconds()))
    return best, stamp(best)


def _read_grib(blob: bytes):
    import pygrib

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(gzip.decompress(blob))
        path = f.name
    try:
        grbs = pygrib.open(path)
        try:
            msg = grbs.message(1)
            return np.asarray(msg.values, dtype=np.float32), msg.latlons()
        finally:
            grbs.close()
    finally:
        os.unlink(path)


def _regrid(vals, covered, src_lat, src_lon, grid):
    """Nearest-neighbour MRMS (~1 km) onto the target grid, carrying its coverage.

    Nearest rather than area-mean on purpose: this is the *truth* field, and
    area-averaging radar would smooth away exactly the small-scale intensity the
    high thresholds are meant to test, flattering the forecast. Where the target is
    near the source resolution (2 km vs 1 km) the two agree anyway.
    """
    lat_idx = np.abs(grid.lat[:, None] - src_lat[None, :]).argmin(axis=1)
    lon_idx = np.abs(grid.lon[:, None] - src_lon[None, :]).argmin(axis=1)
    box = ((grid.lat >= src_lat.min()) & (grid.lat <= src_lat.max()))[:, None] & \
          ((grid.lon >= src_lon.min()) & (grid.lon <= src_lon.max()))[None, :]
    out = vals[np.ix_(lat_idx, lon_idx)].astype(np.float32)
    mask = covered[np.ix_(lat_idx, lon_idx)] & box
    return np.where(mask, out, FILL), mask


def fetch(session: requests.Session, when: datetime, grid=None):
    """MRMS nearest `when` as dBZ on `grid`, plus its coverage mask.

    Returns (dbz, mask, actual_time) or None if nothing close enough exists.
    """
    grid = grid or obs.CONUS_2KM
    try:
        key, stamp = closest_key(session, when)
        if abs((stamp - when).total_seconds()) > MAX_AGE_S:
            return None
        r = session.get(f"{BUCKET}/{key}", timeout=180)
        r.raise_for_status()
        vals, (lats, lons) = _read_grib(r.content)
    except Exception as e:  # noqa: BLE001
        print(f"  mrms {when:%H:%M}: {e}", file=sys.stderr)
        return None

    # MRMS distinguishes the two, and conflating them is the easiest way to get a
    # flattering score: -999 means no radar can see this cell (about a third of the
    # CONUS box, mostly ocean and mountain gaps), while -99 means a radar looked and
    # found nothing. Only the latter is a real dry observation. Scoring the former as
    # dry would pile up correct negatives and lift every model equally.
    covered = vals != -999.0
    vals = np.where(vals < FILL, FILL, vals)  # -99 (no echo) -> the usual floor
    lon1d = lons[0]
    lon1d = np.where(lon1d > 180, lon1d - 360, lon1d)
    dbz, mask = _regrid(vals, covered, lats[:, 0], lon1d, grid)
    return dbz, mask, stamp


if __name__ == "__main__":  # smoke test: python verify/radar.py
    import time

    s = requests.Session()
    t = time.time()
    got = fetch(s, datetime.now(timezone.utc), obs.CONUS_2KM)
    if got is None:
        sys.exit("no MRMS available")
    dbz, mask, stamp = got
    print(f"MRMS {stamp:%Y-%m-%d %H:%M:%SZ} on {obs.CONUS_2KM} in {time.time()-t:.1f}s")
    print(f"  covered {mask.mean():.3f}  wet(>=20dBZ) {(dbz >= 20).mean():.4f}  "
          f"max {dbz.max():.1f} dBZ")
