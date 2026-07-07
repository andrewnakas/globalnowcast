"""Fetch GFS composite reflectivity (REFC) from AWS Open Data via byte-range requests."""
import math
from datetime import datetime, timedelta

import requests

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
MAX_HOURLY_FH = 120  # GFS hourly output ends at f120


def _key(cycle: datetime, fh: int) -> str:
    return (
        f"gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
        f"gfs.t{cycle:%H}z.pgrb2.0p25.f{fh:03d}"
    )


def lead_offset(cycle: datetime, now: datetime) -> int:
    """First forecast hour whose valid time is at or after `now`."""
    return max(0, math.ceil((now - cycle).total_seconds() / 3600))


def find_latest_cycle(session: requests.Session, now: datetime, horizon: int) -> datetime:
    """Newest 6-hourly GFS cycle published out to `horizon` hours past `now`."""
    base = now.replace(minute=0, second=0, microsecond=0)
    base = base.replace(hour=base.hour - base.hour % 6)
    for back in range(9):
        cycle = base - timedelta(hours=6 * back)
        fh = lead_offset(cycle, now) + horizon
        if fh > MAX_HOURLY_FH:
            continue
        r = session.head(f"{BUCKET}/{_key(cycle, fh)}.idx", timeout=30)
        if r.status_code == 200:
            return cycle
    raise RuntimeError("no complete GFS cycle found in the last 48 hours")


def fetch_refc(session: requests.Session, cycle: datetime, fh: int) -> bytes:
    """Download just the REFC GRIB message for one forecast hour (~1 MB)."""
    url = f"{BUCKET}/{_key(cycle, fh)}"
    idx = session.get(f"{url}.idx", timeout=60)
    idx.raise_for_status()
    lines = idx.text.splitlines()
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) > 4 and parts[3] == "REFC":
            start = int(parts[1])
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
            r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            r.raise_for_status()
            return r.content
    raise RuntimeError(f"REFC not found in index for f{fh:03d}")
