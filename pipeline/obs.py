"""Observed precipitation from the NOAA blended geostationary rain-rate product.

Source: `noaa-enterprise-rainrate-pds` (anonymous S3), product RRQPE-INST-GLB — a
global blend of GOES-19, GOES-18, Himawari-9, MSG2 and MSG3 at 0.02 degrees on a
10-minute cadence, roughly 13-22 minutes behind real time. It covers 70N..60S,
about 90% of the globe by area; the polar caps have no observations at all.

Nothing here may raise into the hourly build: every entry point returns None on
failure so a satellite outage degrades the site to GFS-only rather than breaking it.
"""
import io
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

BUCKET = "https://noaa-enterprise-rainrate-pds.s3.amazonaws.com"
PREFIX = "BLEND/RainRate-Blend-INST"

# The trailing -N of a key is a completeness index: N=5 carries every satellite
# (100% of the domain valid) and lands ~22 min after the slot; lower N are earlier
# partial passes, typically ~50% valid. Highest available wins.
GLB_LEVELS = (5, 4, 3, 2)
MIN_PAIR_LEVEL = 4  # both frames of a flow pair must be at least this complete

CADENCE_MIN = 10
LATENCY_MIN = 12  # don't bother asking for slots newer than this
SEARCH_SLOTS = 12  # walk back at most 2 h looking for a usable frame

# Source grid, from the file's geospatial_* attributes. The Rows/Columns datasets
# in the netCDF are empty dimension scales, so the axes must be derived, not read.
SRC_ROWS, SRC_COLS = 6501, 18000
SRC_LAT_MAX, SRC_LAT_MIN = 70.0, -60.0
SRC_LON_MIN, SRC_LON_MAX = -180.0, 179.98
SRC_LAT = np.linspace(SRC_LAT_MAX, SRC_LAT_MIN, SRC_ROWS)
SRC_LON = np.linspace(SRC_LON_MIN, SRC_LON_MAX, SRC_COLS)

# Target grid: identical convention to pipeline/render.decode_refc — row 0 = 90N,
# column 0 = -180 (i.e. post-np.roll).
GFS_LAT = np.linspace(90.0, -90.0, 721)
GFS_LON = np.linspace(-180.0, 179.75, 1440)
FILL = -30.0  # "no echo"; matches correct.DBZ_MIN and ml/data/common.FILL

MIN_VALID_FRAC = 0.5  # a target cell needs this much real data to count as observed
DEADLINE_S = 60.0  # hard cap on the whole obs path


def _reduceat_index(src_axis: np.ndarray, dst_axis: np.ndarray):
    """Start offsets grouping `src_axis` points into `dst_axis` cells, plus counts.

    0.25 / 0.02 = 12.5, so cells alternate between 12 and 13 source pixels. The
    counts are therefore computed rather than assumed — an off-by-one here would
    silently shift the entire field by a fraction of a cell.
    """
    nearest = np.abs(src_axis[:, None] - dst_axis[None, :]).argmin(axis=1)
    dst_hit = np.unique(nearest)
    starts = np.searchsorted(nearest, dst_hit)
    counts = np.add.reduceat(np.ones(src_axis.size, np.float32), starts)
    assert counts.sum() == src_axis.size, "regrid indices lost source pixels"
    return dst_hit, starts, counts


_ROWS, _ROW_STARTS, _ROW_COUNTS = _reduceat_index(SRC_LAT, GFS_LAT)
_COLS, _COL_STARTS, _COL_COUNTS = _reduceat_index(SRC_LON, GFS_LON)

# Which GFS rows the satellites can actually see (521 of 721).
OBS_ROWS = np.zeros(GFS_LAT.size, bool)
OBS_ROWS[_ROWS] = True


def area_mean_regrid(rate: np.ndarray, valid: np.ndarray):
    """Area-mean 0.02deg rain rate onto the 0.25deg GFS grid.

    Nearest-neighbour would throw away ~99% of the source pixels here (each target
    cell covers about 12x12 of them), which measurably distorts the wet fraction.
    Averaging happens in rain rate, never in dBZ: dBZ is logarithmic, so a mean of
    dBZ understates the true cell-mean rate.

    Returns (mm/h, observed-mask) on the full 721x1440 grid; cells outside the
    satellite domain or with too little valid data are 0.0 and False.
    """
    def _block_sum(a):
        return np.add.reduceat(np.add.reduceat(a, _ROW_STARTS, axis=0),
                               _COL_STARTS, axis=1)

    wet = np.where(valid, rate, 0.0).astype(np.float32)
    total = _block_sum(wet)
    seen = _block_sum(valid.astype(np.float32))
    cells = _ROW_COUNTS[:, None] * _COL_COUNTS[None, :]

    frac = seen / cells
    ok = frac >= MIN_VALID_FRAC
    mean = np.divide(total, seen, out=np.zeros_like(total), where=seen > 0)

    out = np.zeros((GFS_LAT.size, GFS_LON.size), np.float32)
    mask = np.zeros((GFS_LAT.size, GFS_LON.size), bool)
    out[np.ix_(_ROWS, _COLS)] = np.where(ok, mean, 0.0)
    mask[np.ix_(_ROWS, _COLS)] = ok
    return out, mask


def rain_to_dbz(rate_mm_hr: np.ndarray) -> np.ndarray:
    """Marshall-Palmer Z = 200 R^1.6, dBZ = 10 log10(Z). Rate <= 0 -> FILL.

    Kept identical to ml/data/common.rain_to_dbz so observations, training targets
    and the blend all share one definition of dBZ.
    """
    r = np.asarray(rate_mm_hr, dtype=np.float64)
    out = np.full(r.shape, FILL, dtype=np.float32)
    wet = r > 0.01
    out[wet] = (10.0 * np.log10(200.0 * np.power(r[wet], 1.6))).astype(np.float32)
    return np.maximum(out, FILL)


def dbz_to_rain(dbz: np.ndarray) -> np.ndarray:
    """Inverse of rain_to_dbz. Values at or below FILL are dry."""
    d = np.asarray(dbz, dtype=np.float32)
    out = np.zeros(d.shape, np.float32)
    wet = d > FILL
    out[wet] = np.power(np.power(10.0, d[wet] / 10.0) / 200.0, 1.0 / 1.6)
    return out


def _slot_of(when: datetime) -> datetime:
    return when.replace(minute=(when.minute // CADENCE_MIN) * CADENCE_MIN,
                        second=0, microsecond=0)


def list_slot(session: requests.Session, slot: datetime) -> dict[int, str]:
    """Available {completeness level: key} for one 10-minute slot."""
    prefix = f"{PREFIX}/{slot:%Y/%m/%d/%H}/"
    r = session.get(f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=200", timeout=30)
    r.raise_for_status()
    stamp = f"_s{slot:%Y%m%d%H%M}"
    found: dict[int, str] = {}
    for node in ET.fromstring(r.content):
        if not node.tag.endswith("Contents"):
            continue
        key = next(c.text for c in node if c.tag.endswith("Key"))
        if stamp not in key:
            continue
        marker = "RRQPE-INST-GLB-"
        i = key.find(marker)
        if i < 0:
            continue
        try:
            found[int(key[i + len(marker)])] = key
        except ValueError:
            continue
    return found


def fetch_key(session: requests.Session, key: str):
    """Download one product file -> (dBZ on the GFS grid, observed mask)."""
    import h5py  # deferred: only the obs path needs it

    r = session.get(f"{BUCKET}/{key}", timeout=(10, 90))
    r.raise_for_status()
    with h5py.File(io.BytesIO(r.content), "r") as f:
        ds = f["RRQPE"]
        raw = ds[:]
        # Mask on the raw int16 before scaling: -9990 * 0.1 compared as a float is
        # a much more fragile test than the exact integer sentinel.
        fill = int(ds.attrs.get("_FillValue", [-9990])[0])
        scale = float(ds.attrs.get("scale_factor", [0.1])[0])
        if raw.shape != (SRC_ROWS, SRC_COLS):
            raise ValueError(f"unexpected RRQPE shape {raw.shape}")
        valid = raw != fill
        rate = raw.astype(np.float32) * scale

    mean_rate, mask = area_mean_regrid(rate, valid)
    dbz = np.where(mask, rain_to_dbz(mean_rate), FILL).astype(np.float32)
    return dbz, mask


def fetch_frame(session: requests.Session, when: datetime, levels=GLB_LEVELS,
                deadline: float | None = None, latency_min: int = LATENCY_MIN):
    """Newest usable observation at or before `when`.

    `latency_min` backs the search off from the requested time; it exists because
    the newest slot is not published yet. When targeting a specific historical slot
    (the older half of a flow pair) pass 0, otherwise the search starts a slot or
    two too early and overshoots the requested separation.

    Returns (dbz, mask, valid_time, level) or None.
    """
    slot = _slot_of(when - timedelta(minutes=latency_min))
    for _ in range(SEARCH_SLOTS):
        if deadline and time.monotonic() > deadline:
            return None
        try:
            available = list_slot(session, slot)
            for level in levels:
                if level not in available:
                    continue
                dbz, mask = fetch_key(session, available[level])
                return dbz, mask, slot, level
        except Exception as e:  # noqa: BLE001 - obs are optional, keep walking back
            print(f"obs: slot {slot:%Y-%m-%d %H:%M} failed: {e}", file=sys.stderr)
        slot -= timedelta(minutes=CADENCE_MIN)
    return None


def latest_pair(session: requests.Session, now: datetime, gap_min: int = 30):
    """Two observations ~`gap_min` apart for motion estimation.

    The separation matters: at 0.25 degrees a 10-minute gap is sub-pixel motion and
    optical flow gains almost nothing over persistence, while ~30 minutes resolves
    several pixels of displacement.

    Both frames must be at least MIN_PAIR_LEVEL complete. Mixing a full frame with a
    half-empty one turns each missing satellite's edge into apparent motion, which
    poisons the flow field exactly where the seams are.

    Returns (prev_dbz, last_dbz, mask, last_time, actual_gap_min) or None.
    """
    deadline = time.monotonic() + DEADLINE_S
    levels = tuple(n for n in GLB_LEVELS if n >= MIN_PAIR_LEVEL)

    last = fetch_frame(session, now, levels=levels, deadline=deadline)
    if last is None:
        return None
    last_dbz, last_mask, last_time, level = last

    # Same level for both halves of the pair, so the visible domain is identical,
    # and target the slot exactly (latency_min=0) rather than backing off again.
    prev = fetch_frame(session, last_time - timedelta(minutes=gap_min),
                       levels=(level,), deadline=deadline, latency_min=0)
    if prev is None:
        return None
    prev_dbz, prev_mask, prev_time, _ = prev

    gap = (last_time - prev_time).total_seconds() / 60.0
    if gap <= 0:
        return None
    return prev_dbz, last_dbz, last_mask & prev_mask, last_time, gap


if __name__ == "__main__":  # smoke test: python pipeline/obs.py
    t0 = time.time()
    s = requests.Session()
    pair = latest_pair(s, datetime.now(timezone.utc))
    if pair is None:
        sys.exit("no observations available")
    prev, last, mask, when, gap = pair
    wet = last > FILL
    print(f"obs at {when:%Y-%m-%d %H:%MZ}, pair gap {gap:.0f} min, {time.time()-t0:.1f}s")
    print(f"  rows observed : {mask.any(axis=1).sum()}/721")
    print(f"  wet fraction  : {wet.mean():.4f}  max {last.max():.1f} dBZ")
    print(f"  prev wet frac : {(prev > FILL).mean():.4f}")
