"""Harvest (GFS REFC forecast, observed rain) pairs for the 6-48h correction model.

No archive carries GFS REFC - dynamical's GFS stores (analysis and forecast) have
precipitation but not reflectivity, and the raw S3 bucket retains ~2 weeks. But the
served product IS REFC, and this session's repeated lesson is to train on exactly
what is served. So pairs are harvested backward from the live bucket: for a valid
time V within the retention window, the RRQPE observation at V exists AND every GFS
cycle from 6 to 48 hours before V is still downloadable. One pass over the window
yields every lead for ~50 valid times; a scheduled re-run grows the set forward.

    python ml/data/build_gfs_pairs.py --days 13 --out ml/gfs_pairs

Each valid time becomes one npz: obs dBZ (721x1440 u8), the observation mask, and
one GFS dBZ field per lead. A shard is skipped if it already exists, so re-runs
only fetch what is new.
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "pipeline"))

import gfs  # noqa: E402
import obs  # noqa: E402
from render import decode_refc  # noqa: E402

U8_LO, U8_HI = -30.0, 60.0
LEADS = (6, 12, 18, 24, 30, 36, 42, 48)


def to_u8(dbz):
    return np.clip((dbz - U8_LO) * (255.0 / (U8_HI - U8_LO)), 0, 255).astype(np.uint8)


def one_valid(session, valid, out_dir):
    path = out_dir / f"{valid:%Y%m%d%H}.npz"
    if path.exists():
        return "cached"
    frame = obs.fetch_frame(session, valid, levels=(5,), latency_min=0)
    if frame is None:
        return "no obs"
    obs_dbz, mask, when, _ = frame
    if abs((when - valid).total_seconds()) > 900:
        return "obs too far"

    fields = {}
    for lead in LEADS:
        cycle = valid - timedelta(hours=lead)
        if cycle.hour % 6:
            continue
        try:
            dbz = np.maximum(decode_refc(gfs.fetch_refc(session, cycle, lead)),
                             obs.FILL)
        except Exception as e:  # noqa: BLE001 - a rolled-off cycle is expected
            print(f"  {valid:%m-%d %HZ} f{lead:03d}: {e}", file=sys.stderr)
            continue
        fields[f"gfs_{lead:03d}"] = to_u8(dbz.astype(np.float32))
    if len(fields) < len(LEADS) - 2:
        return f"only {len(fields)} leads"

    np.savez_compressed(path, obs=to_u8(obs_dbz), mask=mask, **fields)
    return f"ok ({len(fields)} leads)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=13)
    ap.add_argument("--every", type=int, default=6, help="hours between valid times")
    ap.add_argument("--out", default="ml/gfs_pairs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    now = datetime.now(timezone.utc)
    # Newest valid must have its 48h-lead cycle still in the bucket; oldest must
    # itself still be there. The 6h grid keeps every lead's cycle on 00/06/12/18Z.
    latest = now.replace(minute=0, second=0, microsecond=0)
    latest -= timedelta(hours=latest.hour % args.every + args.every)
    kept = 0
    for i in range(args.days * 24 // args.every):
        valid = latest - timedelta(hours=args.every * i)
        status = one_valid(session, valid, out)
        kept += status.startswith(("ok", "cached"))
        print(f"{valid:%Y-%m-%d %HZ}: {status}", flush=True)

    mb = sum(p.stat().st_size for p in out.glob("*.npz")) / 1e6
    print(f"\n{kept} valid times on disk, {mb:.0f} MB in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
