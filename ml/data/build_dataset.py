"""Pair GFS REFC analyses with observed dBZ (IMERG global and/or MRMS CONUS) into shards.

Each sample is (gfs_dbz, obs_dbz) on the GFS 0.25 grid at the same valid time. Shards
are compressed .npz files of float16 arrays — about a year of 6-hourly global pairs is
a few GB. Run this on a machine with the GFS/obs access (locally or in the Kaggle notebook).

Examples:
  # tiny smoke test, MRMS only (no Earthdata token needed):
  python build_dataset.py --source mrms --start 2024-05-20 --days 1 --every 12 --limit 2 --out shards/
  # a year of global IMERG pairs, 6-hourly:
  EARTHDATA_TOKEN=... python build_dataset.py --source imerg --start 2023-06-01 --days 365 --every 6 --out shards/
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imerg  # noqa: E402
import mrms  # noqa: E402
from gfs_label import fetch_analysis  # noqa: E402


def times(start: datetime, days: int, every: int):
    t, end = start, start + timedelta(days=days)
    while t < end:
        yield t
        t += timedelta(hours=every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["imerg", "mrms"], required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--every", type=int, default=6, help="hours between samples (multiple of 6)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of samples (0 = no cap)")
    ap.add_argument("--shard-size", type=int, default=64)
    ap.add_argument("--out", default="shards")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    gfs_buf, obs_buf, shard, made = [], [], 0, 0
    for when in times(start, args.days, args.every):
        if when.hour % 6 != 0:  # GFS analyses only at 00/06/12/18Z
            continue
        try:
            gfs = fetch_analysis(when, session)
            obs = imerg.fetch(when) if args.source == "imerg" else mrms.fetch(when, session)
        except Exception as e:  # noqa: BLE001 - skip a bad timestep, keep going
            print(f"{when:%Y-%m-%d %HZ}: skip ({e})", file=sys.stderr)
            continue
        gfs_buf.append(gfs.astype(np.float16))
        obs_buf.append(obs.astype(np.float16))
        made += 1
        print(f"{when:%Y-%m-%d %HZ}: paired ({made})")

        if len(gfs_buf) >= args.shard_size:
            _flush(outdir, args.source, shard, gfs_buf, obs_buf)
            shard += 1
            gfs_buf, obs_buf = [], []
        if args.limit and made >= args.limit:
            break

    if gfs_buf:
        _flush(outdir, args.source, shard, gfs_buf, obs_buf)
    print(f"done: {made} pairs")


def _flush(outdir, source, shard, gfs_buf, obs_buf):
    path = outdir / f"{source}_{shard:04d}.npz"
    np.savez_compressed(path, gfs=np.stack(gfs_buf), obs=np.stack(obs_buf))
    print(f"wrote {path} ({len(gfs_buf)} pairs)")


if __name__ == "__main__":
    main()
