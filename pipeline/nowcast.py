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
import numpy as np

from obs import FILL, GFS_LAT, GFS_LON, dbz_to_rain, rain_to_dbz

H, W = GFS_LAT.size, GFS_LON.size

# Farneback works on 8-bit images. Feeding it float32 in [0,1] returns a field of
# essentially zeros, which fails silently as a persistence forecast, so the uint8
# conversion below is load-bearing rather than an optimisation.
U8_LO, U8_HI = FILL, 60.0
FLOW_UPSCALE = 2  # estimate motion on a 2x grid to resolve sub-pixel displacement
FLOW_PARAMS = dict(pyr_scale=0.5, levels=5, winsize=31, iterations=3,
                   poly_n=5, poly_sigma=1.2, flags=0)
MAX_FLOW_PX = 20.0  # per 30 min; anything faster means a corrupt frame pair

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
# 330 min is the compromise that scores best across 5/10/20/30 dBZ together.
BLEND_CROSSOVER_MIN = 330.0
BLEND_TAU_MIN = 60.0

# Below this lead the observations are so far ahead of the model that letting any
# GFS in costs more than it adds, so the nowcast is pure extrapolation there.
OBS_ONLY_MIN = 45.0

# Tapering the weight across the 70N/60S boundary was tried and removed: it dilutes
# observations that are perfectly good right up to the edge, and measured a clear
# CSI loss there (0.496 -> 0.447 over the tapered rows) to hide a cosmetic seam.

_GX, _GY = np.meshgrid(np.arange(W + 2 * WRAP_PAD, dtype=np.float32),
                       np.arange(H, dtype=np.float32))


def to_u8(dbz: np.ndarray) -> np.ndarray:
    """dBZ -> uint8 over [FILL, 60]. See the note above: this must not be float."""
    scaled = (np.nan_to_num(dbz, nan=U8_LO) - U8_LO) * (255.0 / (U8_HI - U8_LO))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def estimate_flow(prev_dbz: np.ndarray, last_dbz: np.ndarray,
                  gap_min: float, step_min: float = 30.0) -> np.ndarray:
    """Dense motion field in pixels per `step_min`, shape (H, W, 2) as (dx, dy).

    Estimated on an upscaled pair and divided back down, which recovers motion
    finer than one 0.25 degree cell. The result is rescaled from the pair's actual
    separation to `step_min` so callers advect in consistent units.
    """
    import cv2

    up = lambda a: cv2.resize(to_u8(a), (W * FLOW_UPSCALE, H * FLOW_UPSCALE),
                              interpolation=cv2.INTER_LINEAR)
    flow = cv2.calcOpticalFlowFarneback(up(prev_dbz), up(last_dbz), None, **FLOW_PARAMS)
    flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR) / FLOW_UPSCALE
    flow *= step_min / float(gap_min)

    if np.percentile(np.hypot(flow[..., 0], flow[..., 1]), 99) > MAX_FLOW_PX:
        return np.zeros_like(flow)  # implausible; fall back to persistence
    return flow.astype(np.float32)


def advect(field: np.ndarray, flow: np.ndarray, steps: float) -> np.ndarray:
    """Backward semi-Lagrangian transport: sample where each cell's air came from.

    Fractional `steps` is fine, which is how sub-hourly frames are produced.
    """
    import cv2

    if steps == 0 or not flow.any():
        return field.astype(np.float32)

    src = np.pad(field.astype(np.float32), ((0, 0), (WRAP_PAD, WRAP_PAD)), mode="wrap")
    fl = np.pad(flow, ((0, 0), (WRAP_PAD, WRAP_PAD), (0, 0)), mode="wrap")
    warped = cv2.remap(src,
                       _GX - steps * fl[..., 0],
                       _GY - steps * fl[..., 1],
                       cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=FILL)
    return warped[:, WRAP_PAD:WRAP_PAD + W]


def blend_weight(lead_min: float) -> float:
    """Weight on the observation-based field; GFS takes the remainder.

    Held at exactly 1 for the first stretch, then rescaled so it leaves 1 smoothly
    rather than stepping - a discontinuity here would be visible as a jump in the
    animation.
    """
    if lead_min <= OBS_ONLY_MIN:
        return 1.0
    logistic = lambda t: 1.0 / (1.0 + np.exp((t - BLEND_CROSSOVER_MIN) / BLEND_TAU_MIN))
    # Divide through by the value at the hold point so the curve starts at 1 there.
    return float(min(1.0, logistic(lead_min) / logistic(OBS_ONLY_MIN)))


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
