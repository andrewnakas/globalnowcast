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

    Source points outside the destination axis are dropped, which is what lets a
    sub-global window work: only the pixels inside it are grouped. `full` means the
    axis spans the whole source (the global longitude case), where every point
    belongs somewhere even if it sits past the last cell centre — 179.98 wraps round
    to -180 rather than falling outside the world.
    """
    nearest = np.abs(src_axis[:, None] - dst_axis[None, :]).argmin(axis=1)
    span = abs(dst_axis[-1] - dst_axis[0]) + abs(dst_axis[1] - dst_axis[0])
    covers_all = span >= abs(src_axis[-1] - src_axis[0]) - 1e-9
    if covers_all:
        lo, hi = 0, src_axis.size
    else:
        step = abs(dst_axis[1] - dst_axis[0])
        inside = np.abs(src_axis - dst_axis[nearest]) <= step / 2 + 1e-9
        # Contiguous by construction (both axes monotonic), so a slice suffices.
        lo, hi = int(np.argmax(inside)), src_axis.size - int(np.argmax(inside[::-1]))
        nearest = nearest[lo:hi]
    dst_hit = np.unique(nearest)
    starts = np.searchsorted(nearest, dst_hit)
    counts = np.add.reduceat(np.ones(nearest.size, np.float32), starts)
    assert counts.sum() == nearest.size, "regrid indices lost source pixels"
    return dst_hit, starts, counts, (lo, hi)


class Grid:
    """A target lat/lon grid, plus the regridding plan from the source onto it.

    The global 0.25 degree grid is the default and what the site ships. A window at
    finer spacing is what makes the output comparable to published radar benchmarks,
    which are all regional at 1-2 km; global at 2 km does not fit the runner's memory
    (see ml/PLAN-2KM.md).
    """

    def __init__(self, lat: np.ndarray, lon: np.ndarray, name: str = "global"):
        self.name = name
        self.lat, self.lon = lat, lon
        self.shape = (lat.size, lon.size)
        self.rows, self.row_starts, self.row_counts, self.row_span = \
            _reduceat_index(SRC_LAT, lat)
        self.cols, self.col_starts, self.col_counts, self.col_span = \
            _reduceat_index(SRC_LON, lon)
        # Which target rows the satellites can actually see (521 of 721 globally).
        self.observed_rows = np.zeros(lat.size, bool)
        self.observed_rows[self.rows] = True

    @classmethod
    def window(cls, lat_max, lat_min, lon_min, lon_max, res, name="window"):
        """A regional grid at `res` degrees, snapped to whole source pixels."""
        lat = np.arange(lat_max, lat_min - res / 2, -res)
        lon = np.arange(lon_min, lon_max + res / 2, res)
        return cls(lat, lon, name)

    def __repr__(self):
        res = abs(self.lat[1] - self.lat[0])
        return (f"<Grid {self.name} {self.shape[0]}x{self.shape[1]} @{res:g}deg "
                f"({res * 111:.1f} km)>")


GLOBAL = Grid(GFS_LAT, GFS_LON, "global")

# Benchmark-comparable windows. CONUS matches the domain DGMR and NowcastNet are
# scored on, and MRMS radar truth is available there (see ml/data/mrms.py).
CONUS_2KM = Grid.window(50.0, 24.0, -125.0, -66.0, 0.02, "conus2km")

# Back-compat: the module-level names the rest of the pipeline already imports.
OBS_ROWS = GLOBAL.observed_rows


def area_mean_regrid(rate: np.ndarray, valid: np.ndarray, grid: "Grid" = None):
    """Area-mean 0.02deg rain rate onto `grid` (global 0.25deg by default).

    Nearest-neighbour would throw away ~99% of the source pixels at 0.25deg (each
    target cell covers about 12x12 of them), which measurably distorts the wet
    fraction. Averaging happens in rain rate, never in dBZ: dBZ is logarithmic, so a
    mean of dBZ understates the true cell-mean rate.

    Returns (mm/h, observed-mask) on the grid; cells outside the satellite domain or
    with too little valid data are 0.0 and False.
    """
    grid = grid or GLOBAL
    r0, r1 = grid.row_span
    c0, c1 = grid.col_span

    def _block_sum(a):
        a = a[r0:r1, c0:c1]
        return np.add.reduceat(np.add.reduceat(a, grid.row_starts, axis=0),
                               grid.col_starts, axis=1)

    wet = np.where(valid, rate, 0.0).astype(np.float32)
    total = _block_sum(wet)
    seen = _block_sum(valid.astype(np.float32))
    cells = grid.row_counts[:, None] * grid.col_counts[None, :]

    frac = seen / cells
    ok = frac >= MIN_VALID_FRAC
    mean = np.divide(total, seen, out=np.zeros_like(total), where=seen > 0)

    out = np.zeros(grid.shape, np.float32)
    mask = np.zeros(grid.shape, bool)
    out[np.ix_(grid.rows, grid.cols)] = np.where(ok, mean, 0.0)
    mask[np.ix_(grid.rows, grid.cols)] = ok
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


def fetch_key(session: requests.Session, key: str, grid: "Grid" = None):
    """Download one product file -> (dBZ on `grid`, observed mask)."""
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

    mean_rate, mask = area_mean_regrid(rate, valid, grid)
    dbz = np.where(mask, rain_to_dbz(mean_rate), FILL).astype(np.float32)
    return dbz, mask


def fetch_frame(session: requests.Session, when: datetime, levels=GLB_LEVELS,
                deadline: float | None = None, latency_min: int = LATENCY_MIN,
                grid: "Grid" = None):
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
                dbz, mask = fetch_key(session, available[level], grid)
                return dbz, mask, slot, level
        except Exception as e:  # noqa: BLE001 - obs are optional, keep walking back
            print(f"obs: slot {slot:%Y-%m-%d %H:%M} failed: {e}", file=sys.stderr)
        slot -= timedelta(minutes=CADENCE_MIN)
    return None


def latest_pair(session: requests.Session, now: datetime, gap_min: int = 30,
                grid: "Grid" = None):
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

    last = fetch_frame(session, now, levels=levels, deadline=deadline, grid=grid)
    if last is None:
        return None
    last_dbz, last_mask, last_time, level = last

    # Same level for both halves of the pair, so the visible domain is identical,
    # and target the slot exactly (latency_min=0) rather than backing off again.
    prev = fetch_frame(session, last_time - timedelta(minutes=gap_min),
                       levels=(level,), deadline=deadline, latency_min=0, grid=grid)
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
