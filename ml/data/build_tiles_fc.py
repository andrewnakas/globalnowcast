"""Rebuild nowcast tiles with the HRRR *forecast* as the forecast channel.

The shipped model trained with the HRRR analysis in its forecast channel and it
shows: served true forecast hours (ml/eval_live_inputs.py), the model degrades its
input beyond +1h - the residual head learned corrections to a field far cleaner
than anything available at build time. This builds the same tile sequences with the
channel the live job actually feeds, from dynamical's noaa-hrrr-forecast-48-hour
archive (2018 -> now, 6-hourly inits, hourly leads, one chunk per init and box).

Same sampling as build_tiles.py with the same seed, so the fine-tune sees the same
kind of data and the month-based val split still holds.

    python ml/data/build_tiles_fc.py --tiles 120 --out ml/tiles_fc
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dynamical as dyn  # noqa: E402
from build_tiles import (IN_FRAMES, MIN_WET_FRAC, MIN_WET_FRAMES,  # noqa: E402
                         OUT_FRAMES, SEQ, _flush, sample_positions, to_u8)


def sequences_from_tile_fc(tile, stride, max_per_tile, dry_keep=0.0, rng=None):
    """Windows with a *forecast* HRRR channel, cached per init.

    `dry_keep` keeps that fraction of windows failing the wet filter. Wet-only
    curation is why the hourly model collapsed at frame scale: it never saw the
    98.5% of a real frame that is dry, and hallucinated there. A mixed diet is
    the untested fix.
    """
    wet = tile.wet_fraction
    starts = []
    for s in range(0, len(tile.dbz) - SEQ, stride):
        tgt = wet[s + IN_FRAMES:s + SEQ]
        if (tgt >= MIN_WET_FRAC).sum() < MIN_WET_FRAMES:
            if not (dry_keep and rng is not None and rng.random() < dry_keep):
                continue
        starts.append(s)
        if len(starts) >= max_per_tile:
            break

    samples = []
    by_init: dict = {}
    for s in starts:
        anchor = tile.times[s + IN_FRAMES - 1]
        targets = tile.times[s + IN_FRAMES:s + SEQ]
        # One archive read serves every sequence sharing the init: leads 1..12
        # arrive together, so consecutive anchors inside a 6-hour window slice
        # the same array instead of re-reading the chunk.
        got = None
        for init, (refc, leads) in by_init.items():
            offs = ((targets - init) / np.timedelta64(1, "h")).astype(int)
            if offs.min() >= leads.min() and offs.max() <= leads.max():
                got = refc[offs - leads.min()]
                break
        if got is None:
            fetched = dyn.hrrr_fc_box(anchor, tile.lat, tile.lon)
            if fetched is None:
                continue
            refc, init, leads = fetched
            by_init[init] = (refc, leads)
            offs = ((targets - init) / np.timedelta64(1, "h")).astype(int)
            if offs.min() < leads.min() or offs.max() > leads.max():
                continue
            got = refc[offs - leads.min()]
        seq = tile.dbz[s:s + SEQ]
        samples.append((to_u8(seq[:IN_FRAMES]), to_u8(seq[IN_FRAMES:]),
                        to_u8(got), tile.times[s + IN_FRAMES]))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=120)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-per-tile", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0,
                    help="keep 0: same tile positions as the training build")
    ap.add_argument("--out", default="ml/tiles_fc")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--dry-keep", type=float, default=0.0,
                    help="fraction of dry windows to keep alongside the wet ones")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x_buf, y_buf, h_buf, t_buf = [], [], [], []
    shard = kept = seen = 0
    t_start = time.time()

    # The forecast archive starts 2018-07-13; windows before it can be skipped
    # from the time axis alone, without paying for the MRMS chunk read.
    t_axis = dyn.open_store("mrms")["time"].values
    fc_start = np.datetime64("2018-07-14")

    for k, (i, j, t0) in enumerate(sample_positions(args.tiles, args.seed), 1):
        if t_axis[t0] < fc_start:
            print(f"[{k}/{args.tiles}] ({i:4d},{j:4d}) pre-archive window, skipped")
            continue
        try:
            tile = dyn.mrms_tile(i, j, t0)
        except Exception as e:  # noqa: BLE001
            print(f"[{k}/{args.tiles}] tile ({i},{j}) failed: {e}", file=sys.stderr)
            continue
        seen += 1
        got = sequences_from_tile_fc(tile, args.stride, args.max_per_tile,
                                     dry_keep=args.dry_keep,
                                     rng=np.random.default_rng(args.seed * 100003 + k))
        for x, y, h, t in got:
            x_buf.append(x)
            y_buf.append(y)
            h_buf.append(h)
            t_buf.append(str(t))
            kept += 1
        rate = kept / max(time.time() - t_start, 1e-9)
        print(f"[{k}/{args.tiles}] ({i:4d},{j:4d}) {str(tile.times[0])[:10]}  "
              f"+{len(got):3d} seq  total {kept}  ({rate:.1f}/s)", flush=True)

        while len(x_buf) >= args.shard_size:
            _flush(out, shard, x_buf, y_buf, h_buf, t_buf, args.shard_size)
            shard += 1

    if x_buf:
        _flush(out, shard, x_buf, y_buf, h_buf, t_buf, len(x_buf))
        shard += 1

    mb = sum(p.stat().st_size for p in out.glob("*.npz")) / 1e6
    print(f"\n{kept} sequences from {seen} tiles -> {shard} shard(s), {mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
