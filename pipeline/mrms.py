"""Live MRMS rain-rate history for the radar model.

The radar model's input frames are Marshall-Palmer dBZ derived from the MRMS
precipitation *rate* (ml/data/dynamical.py reads `precipitation_surface` and converts
with the shared rain_to_dbz), not the native reflectivity mosaic. So this fetches
`PrecipRate_00.00` from the raw 2-minute feed and applies the same conversion:
feeding the model MergedReflectivityQCComposite would hand it a field on MRMS's own
regime-dependent Z-R instead of the one it trained on.

Truth-side verification (verify/radar.py) deliberately stays on the reflectivity
mosaic; this module is the *input* side and has to match training, not the truth.

Nothing here may raise into the hourly build: entry points return None on failure so
a radar outage costs the CONUS layer, never the GFS product.
"""
import gzip
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import obs  # noqa: E402

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCT = "PrecipRate_00.00"
FILL = obs.FILL
IN_FRAMES = 4  # matches ml/nowcast_model.IN_FRAMES
# An input frame more than this far from its nominal hour is a different observation.
MAX_AGE_S = 300


def closest_key(session: requests.Session, when: datetime) -> tuple[str, datetime]:
    """The PrecipRate object nearest `when`, with its actual timestamp."""
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
    """Nearest-neighbour ~1 km rate onto the target grid, carrying its coverage.

    Nearest rather than area-mean, matching verify/radar.py's reasoning: averaging
    would smooth away the small-scale intensity the high thresholds test.
    """
    lat_idx = np.abs(grid.lat[:, None] - src_lat[None, :]).argmin(axis=1)
    lon_idx = np.abs(grid.lon[:, None] - src_lon[None, :]).argmin(axis=1)
    box = ((grid.lat >= src_lat.min()) & (grid.lat <= src_lat.max()))[:, None] & \
          ((grid.lon >= src_lon.min()) & (grid.lon <= src_lon.max()))[None, :]
    out = vals[np.ix_(lat_idx, lon_idx)].astype(np.float32)
    mask = covered[np.ix_(lat_idx, lon_idx)] & box
    return np.where(mask, out, 0.0), mask


def fetch_rate(session: requests.Session, when: datetime, grid=None):
    """MRMS rain rate nearest `when` as MP dBZ on `grid`, plus coverage mask.

    Returns (dbz, mask, actual_time) or None if nothing close enough exists.
    """
    grid = grid or obs.CONUS_2KM
    for attempt in range(3):
        try:
            key, stamp = closest_key(session, when)
            if abs((stamp - when).total_seconds()) > MAX_AGE_S:
                return None
            r = session.get(f"{BUCKET}/{key}", timeout=180)
            r.raise_for_status()
            vals, (lats, lons) = _read_grib(r.content)
            break
        except Exception as e:  # noqa: BLE001 - S3 disconnects are routine; retry
            if attempt == 2:
                print(f"  mrms rate {when:%H:%M}: {e}", file=sys.stderr)
                return None
            import time
            time.sleep(2 * (attempt + 1))

    # PrecipRate uses negative sentinels for cells no radar can see (-3); zero is a
    # real dry observation. Scoring or feeding the former as dry would be wrong the
    # same way it is in verify/radar.py.
    covered = vals >= 0.0
    rate = np.where(covered, vals, 0.0)
    lon1d = lons[0]
    lon1d = np.where(lon1d > 180, lon1d - 360, lon1d)
    rate2, mask = _regrid(rate, covered, lats[:, 0], lon1d, grid)
    dbz = np.where(mask, obs.rain_to_dbz(rate2), FILL).astype(np.float32)
    return dbz, mask, stamp


def latest_anchor(session: requests.Session, now: datetime):
    """Newest top-of-hour with an MRMS frame close enough to anchor the model."""
    hour = now.replace(minute=0, second=0, microsecond=0)
    for back in range(2):
        anchor = hour - timedelta(hours=back)
        try:
            _, stamp = closest_key(session, anchor)
        except Exception as e:  # noqa: BLE001
            print(f"  mrms anchor: {e}", file=sys.stderr)
            return None
        if abs((stamp - anchor).total_seconds()) <= MAX_AGE_S:
            return anchor
    return None


def fetch_history(session: requests.Session, anchor: datetime, grid=None):
    """The model's radar input: IN_FRAMES hourly frames ending at `anchor`.

    Returns (dbz stack (IN_FRAMES, H, W), combined mask, per-frame times) or None.
    """
    grid = grid or obs.CONUS_2KM
    frames, times = [], []
    mask = None
    for h in range(IN_FRAMES - 1, -1, -1):
        got = fetch_rate(session, anchor - timedelta(hours=h), grid)
        if got is None:
            return None
        dbz, m, stamp = got
        frames.append(dbz)
        times.append(stamp)
        mask = m if mask is None else (mask & m)
    return np.stack(frames), mask, times


if __name__ == "__main__":  # smoke test: python pipeline/mrms.py
    import time

    s = requests.Session()
    now = datetime.now(timezone.utc)
    t = time.time()
    anchor = latest_anchor(s, now)
    if anchor is None:
        sys.exit("no recent MRMS frame")
    got = fetch_history(s, anchor)
    if got is None:
        sys.exit("history fetch failed")
    stack, mask, times = got
    print(f"anchor {anchor:%Y-%m-%d %H:%MZ}, {time.time()-t:.1f}s total")
    for f, tm in zip(stack, times):
        print(f"  {tm:%H:%M:%SZ}  wet(>=23dBZ) {(f >= 23).mean():.4f}  "
              f"max {f.max():.1f} dBZ")
    print(f"  covered {mask.mean():.3f}  shape {stack.shape}")
