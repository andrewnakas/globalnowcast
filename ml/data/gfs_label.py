"""Fetch the GFS REFC analysis (f000) for a given cycle time, as a dBZ array on its native grid.

Reuses the byte-range REFC fetch from the production pipeline so the training inputs
match exactly what the live site renders.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

# Reuse pipeline/gfs.py and pipeline/render.py without duplicating GRIB logic.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from gfs import fetch_refc  # noqa: E402
from render import decode_refc  # noqa: E402


def fetch_analysis(when: datetime, session: requests.Session | None = None) -> np.ndarray:
    """GFS REFC analysis (f000) at `when` (a 00/06/12/18Z cycle) -> 721x1440 dBZ."""
    session = session or requests.Session()
    grib = fetch_refc(session, when, 0)
    data = decode_refc(grib)                 # already rolled to -180..180, 90N at row 0
    return np.where(data < -30.0, -30.0, data).astype(np.float32)
