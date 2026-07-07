"""Decode a REFC GRIB message and render it as a transparent colormapped PNG."""
import os
import tempfile

import numpy as np
import pygrib
from PIL import Image

# Classic NWS reflectivity palette, one color per 5 dBZ bin from 5 to 75+.
NWS_COLORS = [
    "#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00",
    "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000",
    "#f800fd", "#9854c6",
]
BINS = np.arange(5, 80, 5)


def _build_lut() -> np.ndarray:
    lut = np.zeros((len(BINS) + 1, 4), dtype=np.uint8)  # index 0: < 5 dBZ, transparent
    for i, color in enumerate(NWS_COLORS):
        lut[i + 1] = tuple(int(color[j:j + 2], 16) for j in (1, 3, 5)) + (255,)
    lut[-1] = lut[-2]  # > 75 dBZ reuses the top bin color
    return lut


LUT = _build_lut()


def decode_refc(grib_bytes: bytes) -> np.ndarray:
    """GRIB message -> 721x1440 dBZ array (row 0 = 90N), lons shifted to -180..180."""
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        path = f.name
    try:
        grbs = pygrib.open(path)
        try:
            data = grbs.message(1).values
        finally:
            grbs.close()
    finally:
        os.unlink(path)
    data = np.ma.filled(data, -999.0)
    return np.roll(data, data.shape[1] // 2, axis=1)


def render_png(data: np.ndarray, path: str) -> None:
    rgba = LUT[np.digitize(data, BINS)]
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)
