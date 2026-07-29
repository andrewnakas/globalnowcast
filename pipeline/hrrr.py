"""Fetch HRRR composite reflectivity forecasts via byte-range requests.

Structurally pipeline/gfs.py pointed at `noaa-hrrr-bdp-pds`, with one real addition:
HRRR is on a 3 km Lambert conformal grid, so landing it on the model's lat/lon grid
needs a projection, not an axis lookup. The forward transform is ~20 lines of
spherical LCC and is validated against the file's own coordinates in the smoke test;
scipy stays out of the hourly job.

The model consumes HRRR as its forecast channel: one frame per output lead, valid at
the same hour the model predicts (ml/nowcast_model.py). It trained on the HRRR
*analysis* as that channel, so serving real forecast hours is a measured skew -
ml/eval_live_inputs.py gates on it before anything ships.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import obs  # noqa: E402

BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
FILL = obs.FILL
OUT_FRAMES = 6  # matches ml/nowcast_model.OUT_FRAMES
MAX_FH = 18     # hourly HRRR cycles carry f00-f18; beyond needs the 6-hourly runs
# Publication lag for the forecast hours we need, used only when simulating a
# historical fetch (live code HEAD-checks the bucket instead). f06-f10 of a cycle
# land roughly 50-60 min after cycle time; 60 is the conservative end.
LATENCY_MIN = 60

_grids: dict[str, tuple] = {}


def _key(cycle: datetime, fh: int) -> str:
    return f"hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z.wrfsfcf{fh:02d}.grib2"


def find_cycle(session: requests.Session, anchor: datetime, max_fh: int,
               avail: datetime | None = None) -> datetime:
    """Newest hourly cycle whose forecast reaches `anchor + max_fh` hours.

    `avail` simulates a historical fetch: cycles published after it are skipped, so
    an evaluation sees exactly what the live job would have seen at that moment.
    """
    base = anchor.replace(minute=0, second=0, microsecond=0)
    for back in range(6):
        cycle = base - timedelta(hours=back)
        if avail is not None and cycle + timedelta(minutes=LATENCY_MIN) > avail:
            continue
        fh = int((anchor - cycle).total_seconds() // 3600) + max_fh
        if fh > MAX_FH:
            continue
        if avail is not None:  # historical: existence in the bucket proves nothing
            return cycle
        r = session.head(f"{BUCKET}/{_key(cycle, fh)}.idx", timeout=30)
        if r.status_code == 200:
            return cycle
    raise RuntimeError("no complete HRRR cycle found in the last 6 hours")


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
    raise RuntimeError(f"REFC not found in index for f{fh:02d}")


def _lcc_xy(lat, lon, p):
    """Spherical Lambert conformal forward transform, in the grid's own metres."""
    R = p["a"]
    lat1, lat2 = np.radians(p["lat_1"]), np.radians(p["lat_2"])
    lat0, lon0 = np.radians(p["lat_0"]), np.radians(p["lon_0"])
    la, lo = np.radians(lat), np.radians(lon)
    if abs(lat1 - lat2) < 1e-9:
        n = np.sin(lat1)
    else:
        n = (np.log(np.cos(lat1) / np.cos(lat2)) /
             np.log(np.tan(np.pi / 4 + lat2 / 2) / np.tan(np.pi / 4 + lat1 / 2)))
    F = np.cos(lat1) * np.tan(np.pi / 4 + lat1 / 2) ** n / n
    rho = R * F / np.tan(np.pi / 4 + la / 2) ** n
    rho0 = R * F / np.tan(np.pi / 4 + lat0 / 2) ** n
    dlon = np.arctan2(np.sin(lo - lon0), np.cos(lo - lon0))  # wrap to +-pi
    return rho * np.sin(n * dlon), rho0 - rho * np.cos(n * dlon)


def _grid_index(msg, grid):
    """(iy, ix, inbounds) mapping the target grid onto the HRRR grid, memoised.

    Derives the origin and spacing from the file's own first points rather than
    hardcoding them, so a grid change upstream fails loudly here instead of
    producing a silently shifted field.
    """
    if grid.name in _grids:
        return _grids[grid.name]
    p = dict(msg.projparams)
    lats, lons = msg.latlons()
    x00, y00 = _lcc_xy(lats[0, 0], lons[0, 0], p)
    x01, y01 = _lcc_xy(lats[0, 1], lons[0, 1], p)
    x10, y10 = _lcc_xy(lats[1, 0], lons[1, 0], p)
    dx, dy = x01 - x00, y10 - y00

    gy, gx = np.meshgrid(grid.lat, grid.lon, indexing="ij")
    X, Y = _lcc_xy(gy, gx, p)
    ix = np.rint((X - x00) / dx).astype(np.int32)
    iy = np.rint((Y - y00) / dy).astype(np.int32)
    ny, nx = lats.shape
    inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
    _grids[grid.name] = (iy, ix, inb)
    return _grids[grid.name]


def decode_refc(grib: bytes, grid=None):
    """One REFC message -> (dBZ on `grid`, inbounds mask)."""
    import os
    import tempfile

    import pygrib

    grid = grid or obs.CONUS_2KM
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib)
        path = f.name
    try:
        grbs = pygrib.open(path)
        try:
            msg = grbs.message(1)
            vals = np.asarray(msg.values, dtype=np.float32)
            iy, ix, inb = _grid_index(msg, grid)
        finally:
            grbs.close()
    finally:
        os.unlink(path)
    out = np.where(inb, vals[iy, ix], FILL)
    # Clip to the range training saw (ml/data/build_tiles.to_u8 quantises to it).
    return np.clip(out, -30.0, 60.0).astype(np.float32), inb


def fetch_forecast(session: requests.Session, anchor: datetime, grid=None,
                   avail: datetime | None = None):
    """The model's forecast channel: OUT_FRAMES frames valid anchor+1h..+6h.

    Returns (dbz stack (OUT_FRAMES, H, W), inbounds mask, cycle) or None.
    """
    grid = grid or obs.CONUS_2KM
    try:
        cycle = find_cycle(session, anchor, OUT_FRAMES, avail=avail)
        offset = int((anchor - cycle).total_seconds() // 3600)
        frames = []
        mask = None
        for lead in range(1, OUT_FRAMES + 1):
            dbz, inb = decode_refc(fetch_refc(session, cycle, offset + lead), grid)
            frames.append(dbz)
            mask = inb if mask is None else (mask & inb)
        return np.stack(frames), mask, cycle
    except Exception as e:  # noqa: BLE001 - the CONUS layer is optional
        print(f"  hrrr {anchor:%H:%M}: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":  # smoke test: python pipeline/hrrr.py
    import os
    import tempfile
    import time

    import pygrib

    s = requests.Session()
    now = datetime.now(timezone.utc)
    anchor = now.replace(minute=0, second=0, microsecond=0)

    t = time.time()
    cycle = find_cycle(s, anchor, OUT_FRAMES)
    grib = fetch_refc(s, cycle, 1)
    print(f"cycle {cycle:%Y-%m-%d %HZ}, f01 message {len(grib)/1e6:.1f} MB, "
          f"{time.time()-t:.1f}s")

    # Validate the analytic projection against the file's own coordinates: every
    # 50th source point must map back to its own indices.
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib)
        path = f.name
    try:
        grbs = pygrib.open(path)
        msg = grbs.message(1)
        p = dict(msg.projparams)
        lats, lons = msg.latlons()
    finally:
        grbs.close()
        os.unlink(path)
    x00, y00 = _lcc_xy(lats[0, 0], lons[0, 0], p)
    x01, _ = _lcc_xy(lats[0, 1], lons[0, 1], p)
    _, y10 = _lcc_xy(lats[1, 0], lons[1, 0], p)
    dx, dy = x01 - x00, y10 - y00
    sub = np.s_[::50, ::50]
    X, Y = _lcc_xy(lats[sub], lons[sub], p)
    ix = (X - x00) / dx
    iy = (Y - y00) / dy
    ti, tj = np.meshgrid(np.arange(lats.shape[0])[::50],
                         np.arange(lats.shape[1])[::50], indexing="ij")
    err = max(np.abs(ix - tj).max(), np.abs(iy - ti).max())
    print(f"projection round-trip: max error {err:.3f} cells "
          f"({'ok' if err < 0.5 else 'FAIL'})")

    t = time.time()
    got = fetch_forecast(s, anchor)
    if got is None:
        sys.exit("forecast fetch failed")
    stack, mask, cycle = got
    print(f"forecast from {cycle:%HZ}: {stack.shape} in {time.time()-t:.1f}s")
    for k, f in enumerate(stack, 1):
        print(f"  +{k}h  wet(>=23dBZ) {(f >= 23).mean():.4f}  max {f.max():.1f} dBZ")
