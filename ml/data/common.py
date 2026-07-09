"""Shared helpers: Z-R conversion and regridding observations onto the GFS 0.25 grid."""
import numpy as np

# GFS 0.25-degree global grid: 721 lats (90N..90S), 1440 lons (-180..180 after roll).
GFS_LAT = np.linspace(90.0, -90.0, 721)
GFS_LON = np.linspace(-180.0, 179.75, 1440)
FILL = -30.0  # dBZ value for "no echo" / missing, matches pipeline.correct.DBZ_MIN


def rain_to_dbz(rate_mm_hr: np.ndarray) -> np.ndarray:
    """Marshall-Palmer Z = 200 R^1.6, dBZ = 10 log10(Z). Rate <= 0 -> FILL."""
    r = np.asarray(rate_mm_hr, dtype=np.float64)
    out = np.full(r.shape, FILL, dtype=np.float32)
    wet = r > 0.01  # below ~0.01 mm/hr is effectively no echo
    z = 200.0 * np.power(r[wet], 1.6)
    out[wet] = (10.0 * np.log10(z)).astype(np.float32)
    return np.maximum(out, FILL)


def regrid_to_gfs(values, src_lat, src_lon):
    """Nearest-neighbour regrid of a source lat/lon field onto the GFS 0.25 grid.

    Simple, dependency-free, and adequate at these resolutions (obs are finer than
    the 0.25 target). src_lat/src_lon are 1-D coordinate axes for `values`.
    """
    lat_idx = np.abs(src_lat[:, None] - GFS_LAT[None, :]).argmin(axis=0)
    lon_idx = np.abs(src_lon[:, None] - GFS_LON[None, :]).argmin(axis=0)
    out = values[np.ix_(lat_idx, lon_idx)]
    return np.asarray(out, dtype=np.float32)
