"""Read MRMS radar and HRRR from the dynamical.org Icechunk catalog.

Anonymous S3 in us-west-2, both archives back to 2014 - about 103k hourly steps each.
That is the whole reason to use this catalog: the raw NOAA buckets keep roughly two
weeks of GFS, so anything needing years of history was impossible before.

The chunking dictates how everything here is written. MRMS is chunked
(648 time, 100 lat, 100 lon), so reading one full field costs ~251 s while reading 648
hours of a single 100x100 tile costs ~1.0 s. Always slice a small window over a long
time span, never a whole map.
"""
import sys
from dataclasses import dataclass

import numpy as np

REGION = "us-west-2"
STORES = {
    "mrms": ("dynamical-noaa-mrms", "noaa-mrms-conus-analysis-hourly/v0.3.0.icechunk"),
    "hrrr": ("dynamical-noaa-hrrr", "noaa-hrrr-analysis/v0.2.0.icechunk"),
    "hrrr_fc": ("dynamical-noaa-hrrr", "noaa-hrrr-forecast-48-hour/v0.1.0.icechunk"),
}
MRMS_RATE = "precipitation_surface"       # kg m-2 s-1, x3600 for mm/h
HRRR_REFL = "composite_reflectivity"      # dBZ
FILL = -30.0
TILE = 100                                # matches the MRMS chunk exactly

_CACHE: dict[str, object] = {}


def open_store(which: str):
    """Open (and memoise) one dataset. `which` is 'mrms' or 'hrrr'."""
    if which not in _CACHE:
        import icechunk
        import xarray as xr

        bucket, prefix = STORES[which]
        storage = icechunk.s3_storage(bucket=bucket, prefix=prefix,
                                      region=REGION, anonymous=True)
        repo = icechunk.Repository.open(storage)
        _CACHE[which] = xr.open_zarr(repo.readonly_session("main").store,
                                     consolidated=False, chunks=None)
    return _CACHE[which]


def rain_to_dbz(rate_mm_hr):
    """Marshall-Palmer, identical to pipeline/obs.py so every path shares one dBZ."""
    r = np.asarray(rate_mm_hr, dtype=np.float64)
    out = np.full(r.shape, FILL, dtype=np.float32)
    wet = r > 0.01
    out[wet] = (10.0 * np.log10(200.0 * np.power(r[wet], 1.6))).astype(np.float32)
    return np.maximum(out, FILL)


@dataclass
class Tile:
    """One chunk-aligned MRMS window: a long hourly series over a small area."""
    dbz: np.ndarray          # (time, TILE, TILE) dBZ
    times: np.ndarray        # datetime64
    lat: np.ndarray
    lon: np.ndarray
    i: int                   # tile origin in the MRMS grid
    j: int

    @property
    def wet_fraction(self):
        return (self.dbz >= 23.0).mean(axis=(1, 2))  # >=1 mm/h per frame


def mrms_tile(i: int, j: int, t0: int, n: int = 648) -> Tile:
    """MRMS tile at grid offset (i, j), `n` hourly steps from index `t0`.

    Keep i, j and t0 on multiples of TILE / 648 to stay chunk-aligned; off-grid
    requests still work but read several chunks and are much slower.
    """
    ds = open_store("mrms")
    sub = ds[MRMS_RATE].isel(time=slice(t0, t0 + n),
                             latitude=slice(i, i + TILE),
                             longitude=slice(j, j + TILE))
    rate = np.nan_to_num(sub.values) * 3600.0
    return Tile(rain_to_dbz(rate), sub.time.values,
                ds.latitude.values[i:i + TILE], ds.longitude.values[j:j + TILE], i, j)


def hrrr_on(lat: np.ndarray, lon: np.ndarray, times, var: str = HRRR_REFL):
    """HRRR `var` sampled onto a lat/lon box, for a contiguous span of `times`.

    HRRR is on a Lambert grid with 2-D coordinates, so this nearest-neighbours via a
    KD-tree built on the surrounding sub-box rather than indexing an axis.

    HRRR chunks are (2160 time, 45 y, 45 x), so the cost is dominated by how many
    chunks are touched, not how many timesteps are asked for: 48 consecutive steps of
    one tile read in ~2.1 s while 4 scattered single steps take ~2.7 s. So this reads
    the whole span between the first and last requested time in one slice and then
    picks out the wanted steps, rather than issuing a request per timestep.
    """
    from scipy.spatial import cKDTree

    ds = open_store("hrrr")
    hla, hlo = ds["latitude"].values, ds["longitude"].values
    pad = 0.5
    sel = ((hla >= lat.min() - pad) & (hla <= lat.max() + pad) &
           (hlo >= lon.min() - pad) & (hlo <= lon.max() + pad))
    if not sel.any():
        return None
    ys, xs = np.where(sel)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

    tree = cKDTree(np.c_[hla[y0:y1 + 1, x0:x1 + 1].ravel(),
                         hlo[y0:y1 + 1, x0:x1 + 1].ravel()])
    gy, gx = np.meshgrid(lat, lon, indexing="ij")
    _, idx = tree.query(np.c_[gy.ravel(), gx.ravel()])

    want = np.asarray(times, dtype="datetime64[ns]")
    all_t = ds.time.values
    pos = np.searchsorted(all_t, want)
    pos = np.clip(pos, 0, all_t.size - 1)
    lo, hi = int(pos.min()), int(pos.max())
    try:
        span = ds[var].isel(time=slice(lo, hi + 1),
                            y=slice(y0, y1 + 1), x=slice(x0, x1 + 1)).values
    except Exception as e:  # noqa: BLE001 - a missing span must not kill a batch
        print(f"  hrrr {want[0]}..{want[-1]}: {e}", file=sys.stderr)
        return None

    out = np.empty((len(want), lat.size, lon.size), np.float32)
    for k, p in enumerate(pos):
        out[k] = span[p - lo].ravel()[idx].reshape(lat.size, lon.size)
    return np.maximum(np.nan_to_num(out, nan=FILL), FILL)


def hrrr_fc_box(anchor, lat, lon, max_lead_h: int = 12):
    """HRRR *forecast* composite reflectivity over a lat/lon box, one init.

    The init is the newest one at least an hour old at `anchor` - the same
    walk-back the live job does, except the archive only has 00/06/12/18Z inits,
    so the training channel is on average *older* than the served one (up to
    fh12 vs fh2-7 live). Harder, not easier: a model robust to this is robust to
    serving.

    Returns (refc (leads, ny, nx) on the box's KD-tree mapping, init, lead_hours)
    or None. Reads leads 1..max_lead_h in one go - they live in a single chunk -
    so callers slice locally instead of re-reading per sequence.
    """
    from scipy.spatial import cKDTree

    ds = open_store("hrrr_fc")
    inits = ds.init_time.values
    cutoff = np.datetime64(anchor, "ns") - np.timedelta64(1, "h")
    ok = inits[inits <= cutoff]
    if not len(ok):
        return None
    init = ok[-1]

    hla, hlo = ds["latitude"].values, ds["longitude"].values
    pad = 0.5
    sel = ((hla >= lat.min() - pad) & (hla <= lat.max() + pad) &
           (hlo >= lon.min() - pad) & (hlo <= lon.max() + pad))
    if not sel.any():
        return None
    ys, xs = np.where(sel)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    tree = cKDTree(np.c_[hla[y0:y1 + 1, x0:x1 + 1].ravel(),
                         hlo[y0:y1 + 1, x0:x1 + 1].ravel()])
    gy, gx = np.meshgrid(lat, lon, indexing="ij")
    _, idx = tree.query(np.c_[gy.ravel(), gx.ravel()])

    leads = np.arange(1, max_lead_h + 1)
    try:
        span = ds["composite_reflectivity"].sel(init_time=init).isel(
            lead_time=leads, y=slice(y0, y1 + 1), x=slice(x0, x1 + 1)).values
    except Exception as e:  # noqa: BLE001
        print(f"  hrrr_fc {init}: {e}", file=sys.stderr)
        return None
    out = np.empty((len(leads), lat.size, lon.size), np.float32)
    for k in range(len(leads)):
        out[k] = span[k].ravel()[idx].reshape(lat.size, lon.size)
    return np.maximum(np.nan_to_num(out, nan=FILL), FILL), init, leads


def grid_size():
    ds = open_store("mrms")
    return ds.sizes["time"], ds.sizes["latitude"], ds.sizes["longitude"]


if __name__ == "__main__":  # smoke test: python ml/data/dynamical.py
    import time

    nt, ny, nx = grid_size()
    print(f"MRMS {nt} hourly steps, {ny}x{nx}")
    t = time.time()
    tile = mrms_tile(1200, 3400, nt - 648)
    print(f"tile read {time.time() - t:.1f}s  {tile.dbz.shape}  "
          f"wettest frame {tile.wet_fraction.max():.3f}")
    k = int(tile.wet_fraction.argmax())
    t = time.time()
    h = hrrr_on(tile.lat, tile.lon, tile.times[k:k + 1])
    print(f"hrrr sample {time.time() - t:.1f}s  max {np.nanmax(h):.1f} dBZ "
          f"vs mrms {tile.dbz[k].max():.1f} dBZ")
