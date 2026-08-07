"""Build the Global Nowcast site data: fetch GFS REFC, render frames, write manifest."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

from correct import correct, is_active
from gfs import fetch_refc, find_latest_cycle, lead_offset
from render import decode_refc, render_png

RAPID_HOURS = 18
EXTENDED_HOURS = 48
SITE_DATA = Path(__file__).resolve().parent.parent / "site" / "data"
FRAMES_DIR = SITE_DATA / "frames"

# Observation-driven product: 0-4 h at 15-minute steps, anchored to the latest
# satellite frame rather than the model cycle.
NOWCAST_HORIZON_MIN = 240
NOWCAST_STEP_MIN = 15
# Only the first few forecast hours are needed to blend against; keeping every
# lead would cost ~200 MB for no benefit.
NOWCAST_GFS_HOURS = 6

# CONUS radar-model overlay: rendered on the 2 km window, drawn above the global
# field. Feather the edge over ~0.75 degrees and fade the layer out over its final
# hour so both its spatial boundary and its exit from the animation are soft.
CONUS_BOUNDS = [[24.0, -125.0], [50.0, -66.0]]
CONUS_FEATHER_PX = 37
CONUS_FADE_FROM_MIN = 300.0
CONUS_HORIZON_MIN = 360.0

# Multi-model mean of GFS and ECMWF AIFS, in rain rate. Measured on 16 harvested
# valid times against RRQPE (ml/eval_aifs_blend.py), pooled over 6-48h leads:
#
#   arm     5dBZ    10dBZ   20dBZ   30dBZ   wet-bias
#   gfs     0.1684  0.1763  0.1912  0.0908  1.45
#   aifs    0.2087  0.2370  0.2241  0.0547  1.39
#   mean    0.1929  0.2316  0.2278  0.0853  1.42
#
# AIFS alone is much better at ordinary rain and far worse at heavy cores (it
# barely exceeds 40 dBZ anywhere), so swapping outright would gut the severe
# signal. The mean keeps most of AIFS's gain at every threshold the map renders
# while giving back only 6% at 30 dBZ, and it holds at every lead from +6 to
# +48h. A per-cell max scored better at 30 dBZ but painted 2.28x the observed
# wet area, which is the double-counting this project has rejected before.
AIFS_MIN_LEAD_H = 6  # inside this the blend owns the frame; leave it raw

# Model disagreement is a real skill signal, but it is NOT rendered, and the
# reason is worth recording so nobody rebuilds this.
#
# Binned over 4.8M forecast-wet cells (ml/eval_spread_skill.py) by relative
# spread |aifs-gfs| / mean, the relationship is clean and monotonic:
#
#   spread bin   hit rate   FAR
#   0.00-0.51      0.401    0.599
#   0.96-1.36      0.342    0.658
#   1.71-2.00      0.239    0.761
#
# Rain forecast where the models agree verifies 68% more often than where they
# disagree. That is genuine information. It just does not survive contact with
# a map: four calibrations (relative spread at 1.36, 1.71, 1.9, then absolute
# dBZ difference at 25) all faded 55-73% of visible pixels, because the two
# models really do disagree about most light rain, and light rain is most of
# what is drawn - 40% of visible pixels sit in the faintest 5-10 dBZ bin.
# Fading the majority of a map communicates nothing, and no threshold fixes
# that without simply hiding the disagreement.
#
# The signal is better exposed as an opt-in layer than baked into the base
# field's alpha. Left unbuilt deliberately rather than shipped half-calibrated.


def _upsample(dbz: np.ndarray) -> np.ndarray:
    """Model field (0.25 deg) onto the observation grid (0.1 deg), in rain rate.

    Purely cosmetic continuity: the timeline runs observation -> blend -> model,
    and rendering the last stretch on a 6x coarser grid made the handover look
    like a different product rather than a later forecast. The `keep` copy the
    blend consumes is deliberately NOT upsampled - the blend does its own
    interpolation against the observation grid, and doing it twice would smooth
    the field for no reason.
    """
    import cv2

    import obs as _obs

    target = _obs.GLOBAL_HI.shape
    if dbz.shape == target:
        return dbz
    rate = cv2.resize(_obs.dbz_to_rain(np.maximum(dbz, _obs.FILL)),
                      (target[1], target[0]), interpolation=cv2.INTER_LINEAR)
    return _obs.rain_to_dbz(rate)


def build_frame(session: requests.Session, cycle: datetime, lead: int,
                keep: dict | None = None, aifs_by_valid: dict | None = None):
    valid = cycle + timedelta(hours=lead)
    name = f"refc_{valid:%Y%m%d%H}.png"
    for attempt in range(3):
        try:
            grib = fetch_refc(session, cycle, lead)
            field = correct(decode_refc(grib), lead_h=lead,
                            blend_frame=keep is not None)
            # Multi-model mean with AIFS past the satellite blend's window.
            # Averaged in rain rate, never dBZ, and only where both models have
            # data - a missing AIFS frame simply leaves GFS alone.
            a = (aifs_by_valid or {}).get(valid)
            alpha = None
            if a is not None and keep is None:
                import obs as _obs

                gfs_dbz = np.maximum(field, _obs.FILL)
                aifs_dbz = np.maximum(a, _obs.FILL)
                field = _obs.rain_to_dbz(
                    0.5 * (_obs.dbz_to_rain(gfs_dbz) + _obs.dbz_to_rain(aifs_dbz)))
            # Render on the same grid as the nowcast frames. The models are
            # native 0.25 and the observation product is 0.1, so shipping them
            # at their native sizes made the animation visibly change character
            # partway through: sharp cellular structure up to +4.5h, then
            # abruptly 6x blockier. Upsampling is not new information, but it
            # stops the grid itself from being the most obvious feature of the
            # forecast. In rain rate, never dBZ - dBZ is logarithmic and
            # interpolating it dims the result.
            render_png(_upsample(field), FRAMES_DIR / name, alpha=alpha)
            if keep is not None:
                # Stash the corrected field for the blend so the nowcast sees
                # exactly what shipped, without a second fetch.
                keep[valid] = field.astype("float32")
            return {"file": name, "valid": valid.strftime("%Y-%m-%dT%H:00Z"),
                    "source": "gfs"}
        except Exception as e:  # noqa: BLE001 - a lost frame must not kill the run
            if attempt == 2:
                print(f"f{lead:03d}: giving up: {e}", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))


def build_nowcast(session: requests.Session, now: datetime, gfs_by_valid: dict):
    """Advect the latest observations forward and blend them into GFS.

    Returns (frames, obs_time) or (None, None). Never raises: observations are a
    bonus, and an outage must leave the GFS products untouched.
    """
    try:
        # Imported here so a missing wheel degrades to GFS-only instead of
        # breaking the whole build at import time.
        import cv2

        import nowcast
        import obs

        cv2.setNumThreads(2)  # the runner has few cores; cv2 is already threaded

        # The observation product renders at 0.1 degrees - the satellite is 0.02
        # natively, so this is real structure, not interpolation. GFS stays on its
        # own 0.25 grid and is upsampled (in rain rate) only where the blend
        # needs it.
        grid = obs.GLOBAL_HI
        km_per_px = 0.1 * 111.0
        pair = obs.latest_pair(session, now, grid=grid)
        if pair is None:
            print("nowcast: no observations available", file=sys.stderr)
            return None, None
        prev_dbz, last_dbz, mask, obs_time, gap_min = pair

        flow = nowcast.estimate_flow(prev_dbz, last_dbz, gap_min,
                                     km_per_px=km_per_px)
        p99 = float(np.percentile(np.hypot(flow[..., 0], flow[..., 1]), 99))
        if p99 < 1.0:
            # Flow this small means the field is barely moving - or that the uint8
            # conversion regressed and advection has silently become persistence.
            print(f"nowcast: flow p99 {p99:.2f}px is suspiciously still",
                  file=sys.stderr)

        def upsample(dbz):
            rate = obs.dbz_to_rain(np.maximum(dbz, obs.FILL))
            hi = cv2.resize(rate, (grid.shape[1], grid.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
            return obs.rain_to_dbz(hi)

        frames = []
        for minute in range(0, NOWCAST_HORIZON_MIN + 1, NOWCAST_STEP_MIN):
            valid = obs_time + timedelta(minutes=minute)
            advected = nowcast.advect(last_dbz, flow, minute / 30.0, wrap=True)
            gfs = nowcast.gfs_at(gfs_by_valid, valid)
            if gfs is None:
                field, source = advected, "obs"
            else:
                field = nowcast.blend(advected, upsample(gfs), mask, minute)
                source = "obs" if nowcast.blend_weight(minute) >= 1.0 else "blend"
            name = f"now_{valid:%Y%m%d%H%M}.png"
            render_png(field, FRAMES_DIR / name)
            frames.append({"file": name,
                           "valid": valid.strftime("%Y-%m-%dT%H:%MZ"),
                           "source": source})
        print(f"nowcast: {len(frames)} frames from {obs_time:%Y-%m-%d %H:%MZ} "
              f"(pair gap {gap_min:.0f} min, flow p99 {p99:.1f}px)")
        return frames, obs_time
    except Exception as e:  # noqa: BLE001 - never break the GFS build
        print(f"nowcast: skipped ({e})", file=sys.stderr)
        return None, None


def build_conus_layer(session: requests.Session, now: datetime, valids: list):
    """CONUS radar+HRRR overlay frames for timeline entries inside its horizon.

    Returns ({valid: png name}, info dict) or ({}, None). Same contract as
    build_nowcast: the layer is a bonus, and no failure here may touch the GFS or
    satellite products.
    """
    try:
        import cv2

        import conus

        got = conus.predict(session, now, valids)
        if got is None:
            return {}, None
        fields, mask, info = got
        anchor = info["anchor"]

        # Spatial feather: distance into the radar-covered area, ramped over
        # ~0.75 degrees, so the layer dissolves into the global field instead of
        # ending in a hard line at the coverage boundary.
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        feather = np.clip(dist / CONUS_FEATHER_PX, 0.0, 1.0).astype(np.float32)

        layers = {}
        for valid, field in sorted(fields.items()):
            minutes = (valid - anchor).total_seconds() / 60.0
            fade = 1.0
            if minutes > CONUS_FADE_FROM_MIN:
                fade = (CONUS_HORIZON_MIN - minutes) / \
                       (CONUS_HORIZON_MIN - CONUS_FADE_FROM_MIN)
            name = f"conus_{valid:%Y%m%d%H%M}.png"
            render_png(field, FRAMES_DIR / name, alpha=feather * fade)
            layers[valid] = name
        if layers:
            print(f"conus layer: {len(layers)} frames from {anchor:%Y-%m-%d %H:%MZ} "
                  f"(HRRR {info['cycle']:%HZ})")
        manifest_info = {"anchor": anchor.strftime("%Y-%m-%dT%H:%MZ"),
                         "hrrr_cycle": info["cycle"].strftime("%Y-%m-%dT%H:00Z"),
                         "bounds": CONUS_BOUNDS, "source": info["source"]}
        return layers, (manifest_info if layers else None)
    except Exception as e:  # noqa: BLE001 - never break the build
        print(f"conus layer: skipped ({e})", file=sys.stderr)
        return {}, None


def main() -> None:
    now = datetime.now(timezone.utc)
    session = requests.Session()
    cycle = find_latest_cycle(session, now, horizon=EXTENDED_HOURS)
    offset = lead_offset(cycle, now)
    leads = list(range(offset, offset + EXTENDED_HOURS + 1))
    print(f"cycle {cycle:%Y-%m-%d %HZ}, leads f{leads[0]:03d}..f{leads[-1]:03d}")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_DIR.glob("*.png"):
        old.unlink()

    keep_before = cycle + timedelta(hours=offset + NOWCAST_GFS_HOURS)
    gfs_by_valid: dict[datetime, np.ndarray] = {}

    # AIFS for the frames past the satellite blend's window. Fetched once, up
    # front, because the store is chunked per init: reading each 6-hourly step
    # once and interpolating costs ~110s for the whole horizon, where a
    # per-frame read would cost that many times over.
    t_aifs = time.time()
    aifs_by_valid: dict[datetime, np.ndarray] = {}
    try:
        import aifs as aifs_mod

        want = [cycle + timedelta(hours=l) for l in leads
                if cycle + timedelta(hours=l) > keep_before]
        got = aifs_mod.fetch(now, want)
        if got:
            aifs_by_valid, aifs_init = got
            print(f"aifs: {len(aifs_by_valid)} frames from {aifs_init:%Y-%m-%d %HZ}")
    except Exception as e:  # noqa: BLE001 - AIFS is a bonus arm, never a blocker
        print(f"aifs: skipped ({e})", file=sys.stderr)
    t_aifs = time.time() - t_aifs

    def one(lead: int):
        valid = cycle + timedelta(hours=lead)
        return build_frame(session, cycle, lead,
                           keep=gfs_by_valid if valid <= keep_before else None,
                           aifs_by_valid=aifs_by_valid)

    t_gfs = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(one, leads))
    t_gfs = time.time() - t_gfs

    # Not done here, deliberately: the model frames still paint ~21% of the map
    # against the observation's ~8%, and probability-matching them to the
    # observed wet area would visually smooth that handover. It would also make
    # the forecast worse. verify/gfs_check.py measured frequency-matching GFS
    # losing in 45/45 cases, because the model misplaces rain rather than
    # merely over-producing it - shrinking its wet area discards hits without
    # fixing placement. The wet-area step at the handover is real forecast
    # disagreement, and hiding it would be a cosmetic lie.

    good = [f for f in frames if f]
    # Gate on the GFS frames only; the nowcast is additive and must not be able to
    # fail the run, nor to rescue one.
    if len(good) < 0.8 * len(leads):
        sys.exit(f"only {len(good)}/{len(leads)} frames built - aborting")

    t_now = time.time()
    nowcast_frames, obs_time = build_nowcast(session, now, gfs_by_valid)
    t_now = time.time() - t_now

    # One seamless 0-48h sequence: the 15-minute satellite nowcast while it lasts,
    # then the hourly GFS frames. The products block stays as-is so an older viewer
    # (or a rollback) keeps working from the same manifest.
    def _parse(s):
        return datetime.strptime(s, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)

    timeline = list(nowcast_frames or [])
    cutoff = _parse(timeline[-1]["valid"]) if timeline else None
    timeline += [f for f in good if cutoff is None or _parse(f["valid"]) > cutoff]

    t_conus = time.time()
    valids = [_parse(e["valid"]) for e in timeline]
    layers, conus_info = build_conus_layer(session, now, valids)
    for entry, valid in zip(timeline, valids):
        if valid in layers:
            entry["conus"] = layers[valid]
    t_conus = time.time() - t_conus

    manifest = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cycle": cycle.strftime("%Y-%m-%dT%H:00Z"),
        "corrected": is_active(),
        "products": {
            "rapid": good[: RAPID_HOURS + 1],
            "extended": good,
        },
        "timeline": timeline,
        "multimodel": bool(aifs_by_valid),
    }
    if conus_info:
        manifest["conus"] = conus_info
    if nowcast_frames:
        manifest["obs_time"] = obs_time.strftime("%Y-%m-%dT%H:%MZ")
        manifest["products"]["nowcast"] = nowcast_frames
    (SITE_DATA / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"built {len(good)}/{len(leads)} frames"
          + (f" + {len(nowcast_frames)} nowcast" if nowcast_frames else "")
          + (f" + {len(layers)} conus" if layers else ""))
    print(f"stage times: aifs {t_aifs:.0f}s, gfs {t_gfs:.0f}s, "
          f"nowcast {t_now:.0f}s, conus {t_conus:.0f}s")


if __name__ == "__main__":
    main()
