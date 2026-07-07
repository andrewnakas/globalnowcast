"""Build the Global Nowcast site data: fetch GFS REFC, render frames, write manifest."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from gfs import fetch_refc, find_latest_cycle, lead_offset
from render import decode_refc, render_png

RAPID_HOURS = 18
EXTENDED_HOURS = 48
SITE_DATA = Path(__file__).resolve().parent.parent / "site" / "data"
FRAMES_DIR = SITE_DATA / "frames"


def build_frame(session: requests.Session, cycle: datetime, lead: int):
    valid = cycle + timedelta(hours=lead)
    name = f"refc_{valid:%Y%m%d%H}.png"
    for attempt in range(3):
        try:
            grib = fetch_refc(session, cycle, lead)
            render_png(decode_refc(grib), FRAMES_DIR / name)
            return {"file": name, "valid": valid.strftime("%Y-%m-%dT%H:00Z")}
        except Exception as e:  # noqa: BLE001 - a lost frame must not kill the run
            if attempt == 2:
                print(f"f{lead:03d}: giving up: {e}", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))


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

    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(lambda lead: build_frame(session, cycle, lead), leads))

    good = [f for f in frames if f]
    if len(good) < 0.8 * len(leads):
        sys.exit(f"only {len(good)}/{len(leads)} frames built - aborting")

    manifest = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cycle": cycle.strftime("%Y-%m-%dT%H:00Z"),
        "products": {
            "rapid": good[: RAPID_HOURS + 1],
            "extended": good,
        },
    }
    (SITE_DATA / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"built {len(good)}/{len(leads)} frames")


if __name__ == "__main__":
    main()
