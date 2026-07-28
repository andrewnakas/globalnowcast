"""Build a sub-hourly radar dataset in the format published benchmarks actually use.

The hourly model is at its ceiling: consecutive truth frames one hour apart agree at
only CSI ~0.40, so most of the field has already changed and there is little left to
predict. Published nowcasting work (DGMR, SEVIR, NowcastNet) runs at 5-10 minute steps
out to 90-180 minutes, where far more of the field persists and the task is genuinely
different.

Our hourly product cannot be compared to any of them, which is the real reason
verify/BENCHMARKS.md concludes no leaderboard applies. This builds the data that would
make a comparison possible.

Source is the raw noaa-mrms-pds bucket at ~2-minute cadence, which verify/radar.py
already reads, rather than the hourly dynamical.org archive. It goes back over 400
days - checked, not assumed.

The cost structure decides the shape of this file. One 2 km CONUS frame takes about
6 seconds to fetch and decode, so pulling a 22-frame sequence per sample would be
2.3 minutes each and hopeless. But a single frame covers all of CONUS and tiles into
377 windows of 100x100, so one sequence of fetches yields hundreds of spatially
distinct samples - about 0.4 s per sample instead of 140.

    python ml/data/build_subhourly.py --sequences 90 --out ml/tiles_10min
"""
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "pipeline"))
sys.path.insert(0, str(HERE.parent.parent / "verify"))

import obs  # noqa: E402
import radar  # noqa: E402

STEP_MIN = 10          # matches the cadence benchmarks report at
IN_FRAMES = 4          # 40 min of history
OUT_FRAMES = 18        # 3 h ahead, covering the +90 min most papers stop at
SEQ = IN_FRAMES + OUT_FRAMES
TILE = 100
U8_LO, U8_HI = -30.0, 60.0
MIN_WET_FRAC = 0.05
MIN_WET_FRAMES = 6     # of the 18 targets


def to_u8(dbz):
    return np.clip((dbz - U8_LO) * (255.0 / (U8_HI - U8_LO)), 0, 255).astype(np.uint8)


def fetch_sequence(session, start, grid):
    """SEQ frames at STEP_MIN spacing, or None if any is missing."""
    frames, masks = [], []
    for k in range(SEQ):
        got = radar.fetch(session, start + timedelta(minutes=k * STEP_MIN), grid)
        if got is None:
            return None
        dbz, mask, _ = got
        frames.append(dbz)
        masks.append(mask)
    return np.stack(frames), np.logical_and.reduce(masks)


def tiles_from(seq, mask):
    """Every rain-bearing 100x100 window of one sequence.

    Radar coverage is patchy at the domain edges, so a tile is only usable where the
    mosaic actually sees the whole window - otherwise 'no coverage' would train the
    model to predict dry.
    """
    out = []
    h, w = seq.shape[-2:]
    for i in range(0, h - TILE + 1, TILE):
        for j in range(0, w - TILE + 1, TILE):
            m = mask[i:i + TILE, j:j + TILE]
            if not m.all():
                continue
            t = seq[:, i:i + TILE, j:j + TILE]
            wet = (t[IN_FRAMES:] >= 23.0).mean(axis=(1, 2))
            if (wet >= MIN_WET_FRAC).sum() < MIN_WET_FRAMES:
                continue
            out.append((to_u8(t[:IN_FRAMES]), to_u8(t[IN_FRAMES:])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", type=int, default=90)
    ap.add_argument("--days-back", type=int, default=400,
                    help="sample start times from the last N days")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ml/tiles_10min")
    ap.add_argument("--shard-size", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = obs.CONUS_2KM
    session = requests.Session()
    rng = np.random.default_rng(args.seed)
    now = datetime.now(timezone.utc)

    x_buf, y_buf, t_buf = [], [], []
    shard = kept = 0
    t_start = time.time()

    for n in range(1, args.sequences + 1):
        # Spread across the archive and across the diurnal cycle: convection is
        # strongly time-of-day dependent, and sampling one hour would bias the set.
        start = now - timedelta(minutes=int(rng.integers(180, args.days_back * 1440)))
        try:
            got = fetch_sequence(session, start, grid)
        except Exception as e:  # noqa: BLE001
            print(f"[{n}/{args.sequences}] {start:%Y-%m-%d %H:%M}: {e}", file=sys.stderr)
            continue
        if got is None:
            print(f"[{n}/{args.sequences}] {start:%Y-%m-%d %H:%M}: incomplete, skipped")
            continue
        seq, mask = got
        samples = tiles_from(seq, mask)
        for x, y in samples:
            x_buf.append(x)
            y_buf.append(y)
            t_buf.append(start.strftime("%Y-%m-%dT%H:%M"))
        kept += len(samples)
        rate = kept / max(time.time() - t_start, 1e-9)
        print(f"[{n}/{args.sequences}] {start:%Y-%m-%d %H:%M}  +{len(samples):3d} tiles"
              f"  total {kept}  ({rate:.1f}/s)")

        while len(x_buf) >= args.shard_size:
            _flush(out, shard, x_buf, y_buf, t_buf, args.shard_size)
            shard += 1

    if x_buf:
        _flush(out, shard, x_buf, y_buf, t_buf, len(x_buf))
        shard += 1

    mb = sum(p.stat().st_size for p in out.glob("*.npz")) / 1e6
    print(f"\n{kept} samples at {STEP_MIN}-minute steps -> {shard} shard(s), {mb:.0f} MB")
    print(f"Format: {IN_FRAMES} in, {OUT_FRAMES} out, "
          f"+{STEP_MIN}..+{OUT_FRAMES * STEP_MIN} min")
    return 0


def _flush(out, shard, x_buf, y_buf, t_buf, n):
    path = out / f"sub_{shard:04d}.npz"
    assert len(x_buf) == len(y_buf) == len(t_buf), "sample arrays out of step"
    np.savez_compressed(path, x=np.stack(x_buf[:n]), y=np.stack(y_buf[:n]),
                        t=np.array(t_buf[:n]))
    print(f"  wrote {path.name}: {n} samples, {path.stat().st_size / 1e6:.0f} MB")
    del x_buf[:n], y_buf[:n], t_buf[:n]


if __name__ == "__main__":
    sys.exit(main())
