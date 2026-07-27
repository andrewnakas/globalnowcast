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


def build_frame(session: requests.Session, cycle: datetime, lead: int,
                keep: dict | None = None):
    valid = cycle + timedelta(hours=lead)
    name = f"refc_{valid:%Y%m%d%H}.png"
    for attempt in range(3):
        try:
            grib = fetch_refc(session, cycle, lead)
            field = correct(decode_refc(grib))
            render_png(field, FRAMES_DIR / name)
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

        pair = obs.latest_pair(session, now)
        if pair is None:
            print("nowcast: no observations available", file=sys.stderr)
            return None, None
        prev_dbz, last_dbz, mask, obs_time, gap_min = pair

        flow = nowcast.estimate_flow(prev_dbz, last_dbz, gap_min)
        p99 = float(np.percentile(np.hypot(flow[..., 0], flow[..., 1]), 99))
        if p99 < 1.0:
            # Flow this small means the field is barely moving - or that the uint8
            # conversion regressed and advection has silently become persistence.
            print(f"nowcast: flow p99 {p99:.2f}px is suspiciously still",
                  file=sys.stderr)

        frames = []
        for minute in range(0, NOWCAST_HORIZON_MIN + 1, NOWCAST_STEP_MIN):
            valid = obs_time + timedelta(minutes=minute)
            advected = nowcast.advect(last_dbz, flow, minute / 30.0)
            gfs = nowcast.gfs_at(gfs_by_valid, valid)
            if gfs is None:
                field, source = advected, "obs"
            else:
                field = nowcast.blend(advected, gfs, mask, minute)
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

    def one(lead: int):
        valid = cycle + timedelta(hours=lead)
        return build_frame(session, cycle, lead,
                           keep=gfs_by_valid if valid <= keep_before else None)

    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(one, leads))

    good = [f for f in frames if f]
    # Gate on the GFS frames only; the nowcast is additive and must not be able to
    # fail the run, nor to rescue one.
    if len(good) < 0.8 * len(leads):
        sys.exit(f"only {len(good)}/{len(leads)} frames built - aborting")

    nowcast_frames, obs_time = build_nowcast(session, now, gfs_by_valid)

    manifest = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cycle": cycle.strftime("%Y-%m-%dT%H:00Z"),
        "corrected": is_active(),
        "products": {
            "rapid": good[: RAPID_HOURS + 1],
            "extended": good,
        },
    }
    if nowcast_frames:
        manifest["obs_time"] = obs_time.strftime("%Y-%m-%dT%H:%MZ")
        manifest["products"]["nowcast"] = nowcast_frames
    (SITE_DATA / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"built {len(good)}/{len(leads)} frames"
          + (f" + {len(nowcast_frames)} nowcast" if nowcast_frames else ""))


if __name__ == "__main__":
    main()
