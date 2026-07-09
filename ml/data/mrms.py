"""Download and read MRMS composite reflectivity (CONUS), as dBZ on the GFS grid.

Bucket noaa-mrms-pds is anonymous (no credentials). Product
MergedReflectivityQCComposite_00.50 is grib2.gz at 1km, native dBZ, from 2020-10-14.
Timestamps are ~2 min apart and not aligned to whole minutes, so we list the day
and pick the object closest to the requested time.
"""
import gzip
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
import pygrib
import requests

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCT = "MergedReflectivityQCComposite_00.50"
FILL = -30.0


def _closest_key(session: requests.Session, when: datetime) -> str:
    prefix = f"CONUS/{PRODUCT}/{when:%Y%m%d}/"
    r = session.get(f"{BUCKET}?list-type=2&prefix={prefix}", timeout=60)
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
    return min(keys, key=lambda k: abs((stamp(k) - ref).total_seconds()))


def fetch(when: datetime, session: requests.Session | None = None) -> np.ndarray:
    """Return the MRMS timestep nearest `when` as dBZ on the GFS grid (721x1440).

    CONUS-only: cells outside the MRMS domain are set to FILL.
    """
    from common import regrid_to_gfs

    session = session or requests.Session()
    key = _closest_key(session, when)
    raw = session.get(f"{BUCKET}/{key}", timeout=180)
    raw.raise_for_status()
    grib_bytes = gzip.decompress(raw.content)
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        path = f.name
    try:
        grbs = pygrib.open(path)
        try:
            msg = grbs.message(1)
            vals = msg.values
            lats, lons = msg.latlons()
        finally:
            grbs.close()
    finally:
        os.unlink(path)
    vals = np.ma.filled(np.asarray(vals, dtype=np.float32), FILL)
    vals = np.where(vals < FILL, FILL, vals)          # MRMS uses -999/-99 for no-coverage
    lon1d = lons[0]
    lon1d = np.where(lon1d > 180, lon1d - 360, lon1d)  # 0..360 -> -180..180
    return regrid_to_gfs(vals, lats[:, 0], lon1d)
