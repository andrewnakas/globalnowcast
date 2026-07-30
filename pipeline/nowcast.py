"""Extrapolate observed precipitation forward and blend it into the GFS forecast.

In the first few hours, advecting the observed field along its own motion beats the
model badly. Measured against later observations (CSI at dBZ>=20, 70N..60S):

    lead    persistence   advection   GFS
    +30m    0.536         0.632       0.195
    +90m    0.323         0.420       0.194
    +150m   0.229         0.307       0.203
    +210m   0.177         0.230       0.193
    +270m   0.143         0.183       0.190

GFS only catches up around +4.5 h, which is where the blend hands over. That is far
later than the 2 h rule of thumb, so the weights stay observation-dominated well
past the usual crossover.
"""
import sys

import numpy as np

from obs import FILL, GFS_LAT, GFS_LON, dbz_to_rain, rain_to_dbz

H, W = GFS_LAT.size, GFS_LON.size

# Farneback works on 8-bit images. Feeding it float32 in [0,1] returns a field of
# essentially zeros, which fails silently as a persistence forecast, so the uint8
# conversion below is load-bearing rather than an optimisation.
U8_LO, U8_HI = FILL, 60.0
FLOW_UPSCALE = 2  # estimate motion on a 2x grid to resolve sub-pixel displacement
UPSCALE_MAX_PX = 4096  # above this the grid is fine enough that upscaling is waste
FLOW_PARAMS = dict(pyr_scale=0.5, levels=5, winsize=31, iterations=3,
                   poly_n=5, poly_sigma=1.2, flags=0)
# Sanity bound on storm motion. Expressed as a speed, not a pixel count: a fixed
# pixel budget means something different on every grid, and a limit tuned for 28 km
# cells silently rejects ordinary motion at 2 km, degrading the nowcast to
# persistence without saying so.
#
# This is a corruption check, not a meteorological one. Measured 99th-percentile
# motion on real global pairs has a median around 210 km/h and reaches well beyond
# that in fast flow, so the bound sits high deliberately; it only needs to catch a
# mismatched or garbled frame pair. 1100 km/h preserves the behaviour of the pixel
# limit this replaces on the global grid.
MAX_FLOW_KMH = 1100.0

# Longitude is periodic but cv2.remap is not, so the field is padded before warping
# and cropped after. Without this the antimeridian grows a dry seam - right where
# GOES-19 and Himawari-9 overlap.
WRAP_PAD = 96

# Logistic handover, centred on the measured advection/GFS crossover. A linear ramp
# would sit near 0.67 at +90 min, badly under-weighting observations just where they
# hold roughly twice the skill.
# Handover lead. The crossover depends on how heavy the rain is: at 20 dBZ the model
# catches up around +4.5 h, but at the light thresholds the map actually renders it
# never does, so handing over early visibly floods the picture with model drizzle.
# Fitted over 14 cases across a week at 5/10/20/30 dBZ: 360 min maximises mean CSI
# and is the shortest handover that never loses to pure advection at any lead or
# threshold. It is a real interior optimum (360 > 390 > 420 > 480).
BLEND_CROSSOVER_MIN = 360.0
BLEND_TAU_MIN = 60.0

# Below this lead the observations are so far ahead of the model that letting any
# GFS in costs more than it adds, so the nowcast is pure extrapolation there.
OBS_ONLY_MIN = 45.0

# Tapering the weight across the 70N/60S boundary was tried and removed: it dilutes
# observations that are perfectly good right up to the edge, and measured a clear
# CSI loss there (0.496 -> 0.447 over the tapered rows) to hide a cosmetic seam.

_MESH: dict[tuple, tuple] = {}


def _mesh(h: int, w: int):
    """Cached pixel-coordinate grids for remap, keyed by shape.

    Built once per shape: at 2 km these are hundreds of MB, so rebuilding them per
    frame would dominate the runtime.
    """
    key = (h, w)
    if key not in _MESH:
        _MESH[key] = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    return _MESH[key]


def to_u8(dbz: np.ndarray) -> np.ndarray:
    """dBZ -> uint8 over [FILL, 60]. See the note above: this must not be float."""
    scaled = (np.nan_to_num(dbz, nan=U8_LO) - U8_LO) * (255.0 / (U8_HI - U8_LO))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def estimate_flow(prev_dbz: np.ndarray, last_dbz: np.ndarray,
                  gap_min: float, step_min: float = 30.0,
                  km_per_px: float = 0.25 * 111.0) -> np.ndarray:
    """Dense motion field in pixels per `step_min`, shape (H, W, 2) as (dx, dy).

    Estimated on an upscaled pair and divided back down, which recovers motion
    finer than one cell on a coarse grid. The result is rescaled from the pair's
    actual separation to `step_min` so callers advect in consistent units.

    `km_per_px` is only used to sanity-check the result against a physical speed;
    it defaults to the global 0.25 degree grid.
    """
    import cv2

    h, w = prev_dbz.shape
    # Upscaling resolves sub-pixel motion, which matters at 0.25 degrees where 10
    # minutes of drift is under a cell. On a fine grid the motion is already several
    # pixels, so the upscale earns nothing and costs 4x the memory.
    up_factor = FLOW_UPSCALE if max(h, w) <= UPSCALE_MAX_PX else 1
    if up_factor > 1:
        up = lambda a: cv2.resize(to_u8(a), (w * up_factor, h * up_factor),
                                  interpolation=cv2.INTER_LINEAR)
    else:
        up = to_u8
    flow = cv2.calcOpticalFlowFarneback(up(prev_dbz), up(last_dbz), None, **FLOW_PARAMS)
    if up_factor > 1:
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / up_factor
    flow *= step_min / float(gap_min)

    p99_px = np.percentile(np.hypot(flow[..., 0], flow[..., 1]), 99)
    kmh = p99_px * km_per_px * (60.0 / step_min)
    if kmh > MAX_FLOW_KMH:
        print(f"nowcast: rejecting flow, p99 {kmh:.0f} km/h exceeds "
              f"{MAX_FLOW_KMH:.0f}", file=sys.stderr)
        return np.zeros_like(flow)  # implausible; fall back to persistence
    return flow.astype(np.float32)


def advect(field: np.ndarray, flow: np.ndarray, steps: float,
           wrap: bool | None = None) -> np.ndarray:
    """Backward semi-Lagrangian transport: sample where each cell's air came from.

    Fractional `steps` is fine, which is how sub-hourly frames are produced.
    `wrap` marks a longitude-periodic grid; by default only the classic 0.25
    global grid wraps, so callers on other global grids must say so - a regional
    window must NOT, or Atlantic weather advects into the Pacific.
    """
    import cv2

    if steps == 0 or not flow.any():
        return field.astype(np.float32)

    h, w = field.shape
    pad = WRAP_PAD if (wrap if wrap is not None else w == W) else 0
    if pad:
        src = np.pad(field.astype(np.float32), ((0, 0), (pad, pad)), mode="wrap")
        fl = np.pad(flow, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
    else:
        src, fl = field.astype(np.float32), flow
    gx, gy = _mesh(h, w + 2 * pad)
    warped = cv2.remap(src,
                       gx - steps * fl[..., 0],
                       gy - steps * fl[..., 1],
                       cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=FILL)
    return warped[:, pad:pad + w] if pad else warped


def blend_weight(lead_min: float, crossover: float = BLEND_CROSSOVER_MIN,
                 tau: float = BLEND_TAU_MIN, hold: float = OBS_ONLY_MIN) -> float:
    """Weight on the observation-based field; the model takes the remainder.

    Held at exactly 1 for the first stretch, then rescaled so it leaves 1 smoothly
    rather than stepping - a discontinuity here would be visible as a jump in the
    animation. The defaults are the satellite/GFS fit; the CONUS radar/HRRR layer
    passes its own much earlier crossover (radar loses to HRRR around +2 h, the
    satellite only loses to GFS around +4.5 h).
    """
    if lead_min <= hold:
        return 1.0
    logistic = lambda t: 1.0 / (1.0 + np.exp((t - crossover) / tau))
    # Divide through by the value at the hold point so the curve starts at 1 there.
    return float(min(1.0, logistic(lead_min) / logistic(hold)))


def blend(adv_dbz: np.ndarray, gfs_dbz: np.ndarray, mask: np.ndarray,
          lead_min: float) -> np.ndarray:
    """Convex blend of advected observations and GFS, in linear rain rate.

    Rain rate rather than dBZ because dBZ is logarithmic: averaging it is a
    geometric mean that systematically dims the result. Measured better at every
    lead.
    """
    # decode_refc fills gaps with -999 while everything here uses -30. Reconciled
    # inside the blend so no caller can forget it and poison the arithmetic.
    gfs = np.maximum(gfs_dbz, FILL).astype(np.float32)

    w = blend_weight(lead_min) * mask
    rate = w * dbz_to_rain(adv_dbz) + (1.0 - w) * dbz_to_rain(gfs)
    out = rain_to_dbz(rate)
    # Outside the observed domain the model is all there is, undamped.
    return np.where(mask, out, gfs).astype(np.float32)


def gfs_at(gfs_by_valid: dict, when) -> np.ndarray | None:
    """GFS field at an arbitrary time, interpolated between bracketing hours.

    Interpolation is in rain rate, for the same reason the blend is: a step-function
    GFS would also make the animation pulse once an hour as each frame snaps over.
    """
    if not gfs_by_valid:
        return None
    times = sorted(gfs_by_valid)
    if when <= times[0]:
        return gfs_by_valid[times[0]]
    if when >= times[-1]:
        return gfs_by_valid[times[-1]]

    hi = next(t for t in times if t >= when)
    lo = max(t for t in times if t <= when)
    if hi == lo:
        return gfs_by_valid[lo]

    f = (when - lo).total_seconds() / (hi - lo).total_seconds()
    a = dbz_to_rain(np.maximum(gfs_by_valid[lo], FILL))
    b = dbz_to_rain(np.maximum(gfs_by_valid[hi], FILL))
    return rain_to_dbz((1.0 - f) * a + f * b)
