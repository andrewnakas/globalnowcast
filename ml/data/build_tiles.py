"""Build a radar-target nowcasting dataset from the dynamical.org catalog.

Each sample is a short sequence over one 100x100 (~111 km) tile:

    inputs   4 hourly MRMS radar frames, plus the HRRR forecast valid at each output
    targets  6 hourly MRMS radar frames

That is the DGMR/SEVIR formulation, with the difference that HRRR is supplied as a
side input. HRRR alone already scores CSI 0.65 against radar at 1 mm/h - three times
what the satellite retrieval manages - so the model is not being asked to invent the
forecast, only to correct a model it can see and to extrapolate the radar it is given.

Two things about the source dictate the shape of this file:

  * MRMS chunks are (648 time, 100 lat, 100 lon). Reading one chunk gets 27 days of a
    single tile in ~1 s; reading one full map costs ~251 s. So sampling is per tile,
    and every sample from a tile comes out of the same read.
  * Most of CONUS is dry most of the time. Sampling uniformly gives a dataset of empty
    sky, and optical flow (or a model) has nothing to learn from it. Sequences are
    kept only when enough of the target frames actually contain rain.

Stored as uint8 over [-30, 60] dBZ: ~100 KB per sample, so ~100k samples is ~10 GB and
fits Kaggle's 20 GB dataset limit.

    python ml/data/build_tiles.py --tiles 200 --out ml/tiles
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dynamical as dyn  # noqa: E402

IN_FRAMES = 4
OUT_FRAMES = 6
SEQ = IN_FRAMES + OUT_FRAMES
U8_LO, U8_HI = -30.0, 60.0
MIN_WET_FRAC = 0.05   # of the target frames, at >=1 mm/h
MIN_WET_FRAMES = 3    # how many targets must clear it


def to_u8(dbz):
    """dBZ -> uint8. Quantise after the dBZ transform, never on the raw rate: a
    typical tile's 99th percentile is ~0.7 mm/h, so scaling rates directly collapses
    almost everything to zero."""
    return np.clip((dbz - U8_LO) * (255.0 / (U8_HI - U8_LO)), 0, 255).astype(np.uint8)


def from_u8(u8):
    return u8.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO


def sample_positions(n_tiles, seed):
    """Chunk-aligned (i, j, t0), away from the mosaic's thin edges."""
    nt, ny, nx = dyn.grid_size()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_tiles):
        i = int(rng.integers(4, (ny - dyn.TILE) // dyn.TILE - 4)) * dyn.TILE
        j = int(rng.integers(4, (nx - dyn.TILE) // dyn.TILE - 4)) * dyn.TILE
        t0 = int(rng.integers(0, max(1, (nt - 648) // 648))) * 648
        out.append((i, j, t0))
    return out


def sequences_from_tile(tile, stride, max_per_tile, with_hrrr):
    """Every wet-enough window in one tile, as (inputs, targets, hrrr, time)."""
    wet = tile.wet_fraction
    out = []
    for s in range(0, len(tile.dbz) - SEQ, stride):
        tgt = wet[s + IN_FRAMES:s + SEQ]
        if (tgt >= MIN_WET_FRAC).sum() < MIN_WET_FRAMES:
            continue
        out.append(s)
        if len(out) >= max_per_tile:
            break
    if not out:
        return []

    hrrr = None
    if with_hrrr:
        # One KD-tree build per tile, reused for every sequence in it.
        want = sorted({s + IN_FRAMES + k for s in out for k in range(OUT_FRAMES)})
        got = dyn.hrrr_on(tile.lat, tile.lon, tile.times[want])
        if got is not None:
            hrrr = {t: got[n] for n, t in enumerate(want)}

    samples = []
    for s in out:
        seq = tile.dbz[s:s + SEQ]
        h = None
        if hrrr is not None:
            h = np.stack([hrrr[s + IN_FRAMES + k] for k in range(OUT_FRAMES)])
        samples.append((to_u8(seq[:IN_FRAMES]), to_u8(seq[IN_FRAMES:]),
                        None if h is None else to_u8(h), tile.times[s + IN_FRAMES]))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=200)
    ap.add_argument("--stride", type=int, default=3, help="hours between windows")
    ap.add_argument("--max-per-tile", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ml/tiles")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--no-hrrr", action="store_true",
                    help="radar only; much faster, useful for a first pass")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x_buf, y_buf, h_buf, t_buf = [], [], [], []
    shard = kept = seen = 0
    t_start = time.time()

    for k, (i, j, t0) in enumerate(sample_positions(args.tiles, args.seed), 1):
        try:
            tile = dyn.mrms_tile(i, j, t0)
        except Exception as e:  # noqa: BLE001 - one bad tile must not end the build
            print(f"[{k}/{args.tiles}] tile ({i},{j}) failed: {e}", file=sys.stderr)
            continue
        seen += 1
        got = sequences_from_tile(tile, args.stride, args.max_per_tile,
                                  not args.no_hrrr)
        for x, y, h, t in got:
            # Drop the sample rather than store a short h: appending to x and y but
            # not h desynchronises the arrays, and every sample after the gap then
            # silently trains against another sample's HRRR.
            if h is None and not args.no_hrrr:
                continue
            x_buf.append(x)
            y_buf.append(y)
            t_buf.append(str(t))
            if h is not None:
                h_buf.append(h)
            kept += 1
        rate = kept / max(time.time() - t_start, 1e-9)
        print(f"[{k}/{args.tiles}] ({i:4d},{j:4d}) {str(tile.times[0])[:10]}  "
              f"+{len(got):3d} seq  total {kept}  ({rate:.1f}/s)")

        while len(x_buf) >= args.shard_size:
            _flush(out, shard, x_buf, y_buf, h_buf, t_buf, args.shard_size)
            shard += 1

    if x_buf:
        _flush(out, shard, x_buf, y_buf, h_buf, t_buf, len(x_buf))
        shard += 1

    mb = sum(p.stat().st_size for p in out.glob("*.npz")) / 1e6
    print(f"\n{kept} sequences from {seen} tiles -> {shard} shard(s), {mb:.0f} MB")
    print(f"held-out months are chosen at train time from the stored timestamps, so "
          f"nothing leaks across time")
    return 0


def _flush(out, shard, x_buf, y_buf, h_buf, t_buf, n):
    path = out / f"tiles_{shard:04d}.npz"
    # Sample i of every array must describe the same sequence. A short h would pair
    # each sample after the gap with another sample's forecast, which trains happily
    # and is invisible in the loss.
    assert len(x_buf) == len(y_buf) == len(t_buf), "sample arrays out of step"
    assert not h_buf or len(h_buf) == len(x_buf), "hrrr array out of step"
    data = {"x": np.stack(x_buf[:n]), "y": np.stack(y_buf[:n]),
            "t": np.array(t_buf[:n])}
    if h_buf:
        data["h"] = np.stack(h_buf[:n])
    np.savez_compressed(path, **data)
    print(f"  wrote {path.name}: {data['x'].shape[0]} seq, "
          f"{path.stat().st_size / 1e6:.0f} MB")
    del x_buf[:n], y_buf[:n], t_buf[:n]
    if h_buf:
        del h_buf[:n]


if __name__ == "__main__":
    sys.exit(main())
