"""Download and read GPM IMERG half-hourly precipitation (global), as dBZ on the GFS grid.

Uses the NASA GES DISC HTTPS server, which needs a free Earthdata Login. Provide a
bearer token via the EARTHDATA_TOKEN env var (create at urs.earthdata.nasa.gov, and
authorize the "NASA GESDISC DATA ARCHIVE" application in your profile). Tokens last ~60 days.
"""
import io
import os
from datetime import datetime

import h5py
import numpy as np
import requests

from common import rain_to_dbz, regrid_to_gfs

BASE = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3"
RUN = {"late": ("GPM_3IMERGHHL.07", "L"), "early": ("GPM_3IMERGHHE.07", "E")}


def _url(when: datetime, run: str) -> str:
    """IMERG slices are aligned to :00 and :30; `when` is snapped to the slice start."""
    coll, tag = RUN[run]
    minute = 0 if when.minute < 30 else 30
    start_min = when.hour * 60 + minute          # minute-of-day, the 4-digit field
    s = f"{when.hour:02d}{minute:02d}00"          # S000000 / S003000
    end_total = start_min + 29                    # slice covers 29:59 minutes
    e = f"{end_total // 60:02d}{end_total % 60:02d}59"
    doy = when.timetuple().tm_yday
    fname = (
        f"3B-HHR-{tag}.MS.MRG.3IMERG.{when:%Y%m%d}"
        f"-S{s}-E{e}.{start_min:04d}.V07B.HDF5"
    )
    return f"{BASE}/{coll}/{when:%Y}/{doy:03d}/{fname}"


def fetch(when: datetime, run: str = "late", token: str | None = None) -> np.ndarray:
    """Return an IMERG timestep as dBZ regridded to the GFS 0.25 grid (721x1440)."""
    token = token or os.environ["EARTHDATA_TOKEN"]
    r = requests.get(_url(when, run), headers={"Authorization": f"Bearer {token}"}, timeout=180)
    r.raise_for_status()
    with h5py.File(io.BytesIO(r.content), "r") as f:
        grid = f["Grid"]
        precip = grid["precipitation"][0]          # (lon, lat), mm/hr, fill = -9999.9
        lat = grid["lat"][:]                         # 1800, -90..90
        lon = grid["lon"][:]                         # 3600, -180..180
    precip = np.where(precip < 0, 0.0, precip).T     # -> (lat, lon)
    dbz = rain_to_dbz(precip)
    return regrid_to_gfs(dbz, lat, lon)
