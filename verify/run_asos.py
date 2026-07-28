"""Score the nowcast against rain gauges instead of radar.

Every other number in this project is verified against MRMS, which is itself a radar
retrieval: backscatter aloft converted to a rate through an assumed Z-R relationship.
A model trained on radar and scored against radar shares all of that. Gauges measure
water landing in a bucket, and their errors - undercatch in wind, lag in cold events -
are unrelated to radar's.

So this is not a better verification, it is an independent one. If the model's lead
over HRRR survives here, the radar numbers mean what they appear to. If it shrinks,
part of that lead is agreement with one instrument rather than skill.

    python verify/run_asos.py --hours 48

Two things have to line up or the comparison is meaningless, and both are easy to get
wrong. ASOS `p01m` is precipitation accumulated over the hour *ending* at the
observation time, while the model predicts an instantaneous rate - so the model field
is integrated over that hour before comparing. And a gauge is a point while a model
cell is ~2 km across, so a station is matched to the cell containing it, not
interpolated.
"""
import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))

import obs  # noqa: E402
import radar  # noqa: E402
from metrics import csi  # noqa: E402

ASOS = ("https://data.source.coop/dynamical/asos-parquet/"
        "year={year}/data.parquet")
# Gauge thresholds in mm accumulated over an hour, matching the rate thresholds used
# elsewhere closely enough to be read side by side.
THRESHOLDS = (0.2, 1.0, 4.0, 8.0)


def load_stations(start, end):
    """Hourly gauge reports inside the CONUS grid, as a DataFrame."""
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    years = sorted({start.year, end.year})
    urls = ", ".join(f"'{ASOS.format(year=y)}'" for y in years)
    g = obs.CONUS_2KM
    # One report per station per hour. Stations file METARs plus specials - up to 14
    # an hour at busy airports - and p01m repeats across them, so taking every row
    # would count the same gauge many times and weight big airports over rural ones.
    # The report nearest the top of the hour is the one whose p01m covers that hour.
    return con.execute(f"""
        SELECT station, valid, latitude, longitude, p01m FROM (
            SELECT station, valid, latitude, longitude, p01m,
                   row_number() OVER (
                       PARTITION BY station, date_trunc('hour', valid)
                       ORDER BY abs(epoch(valid - date_trunc('hour', valid)) - 3600)
                   ) AS rn
            FROM read_parquet([{urls}])
            WHERE p01m IS NOT NULL
              AND valid >= TIMESTAMPTZ '{start:%Y-%m-%d %H:%M:%S+00}'
              AND valid <  TIMESTAMPTZ '{end:%Y-%m-%d %H:%M:%S+00}'
              AND latitude BETWEEN {g.lat.min()} AND {g.lat.max()}
              AND longitude BETWEEN {g.lon.min()} AND {g.lon.max()}
        ) WHERE rn = 1
    """).df()


def accumulate(session, grid, end_time, steps=6):
    """Mean rain rate over the hour ending at `end_time`, as mm.

    Radar frames are ~2 minutes apart; sampling every 10 minutes over the hour and
    averaging is a fair approximation of the accumulation a gauge measured, without
    fetching thirty frames per hour.
    """
    rates, mask = [], None
    for k in range(steps):
        t = end_time - timedelta(minutes=10 * k)
        got = radar.fetch(session, t, grid)
        if got is None:
            continue
        dbz, m, _ = got
        rates.append(obs.dbz_to_rain(dbz))
        mask = m if mask is None else (mask & m)
    if not rates:
        return None, None
    return np.mean(rates, axis=0), mask  # mm/h averaged over the hour == mm accumulated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24, help="how many hours to score")
    ap.add_argument("--end-lag", type=int, default=6,
                    help="hours behind now, so both sources are complete")
    args = ap.parse_args()

    import requests

    grid = obs.CONUS_2KM
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) \
        - timedelta(hours=args.end_lag)
    start = end - timedelta(hours=args.hours)
    print(f"{start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} UTC")

    df = load_stations(start, end)
    print(f"{len(df)} gauge reports from {df.station.nunique()} stations")
    if df.empty:
        return 1

    session = requests.Session()
    # Radar cells containing each station, computed once.
    lat_idx = np.abs(grid.lat[None, :] - df.latitude.values[:, None]).argmin(axis=1)
    lon_idx = np.abs(grid.lon[None, :] - df.longitude.values[:, None]).argmin(axis=1)
    df = df.assign(_i=lat_idx, _j=lon_idx)

    pooled = defaultdict(lambda: [0, 0, 0])
    n_hours = 0
    for hour, rows in df.groupby(df.valid.dt.floor("h")):
        t = hour.to_pydatetime().astimezone(timezone.utc)
        acc, mask = accumulate(session, grid, t)
        if acc is None:
            continue
        keep = mask[rows._i.values, rows._j.values]
        if not keep.any():
            continue
        radar_mm = acc[rows._i.values, rows._j.values][keep]
        gauge_mm = rows.p01m.values[keep]
        n_hours += 1
        for thr in THRESHOLDS:
            p, o = radar_mm >= thr, gauge_mm >= thr
            c = pooled[thr]
            c[0] += int((p & o).sum())
            c[1] += int((~p & o).sum())
            c[2] += int((p & ~o).sum())
        print(f"  {t:%m-%d %HZ}: {int(keep.sum())} stations, "
              f"gauge wet {float((gauge_mm >= 0.2).mean()):.3f} "
              f"radar wet {float((radar_mm >= 0.2).mean()):.3f}")

    if not n_hours:
        print("nothing scored")
        return 1

    print(f"\n=== MRMS radar vs ASOS gauges, {n_hours} hour(s) ===")
    print(f"{'threshold':>12}{'CSI':>9}{'POD':>8}{'FAR':>8}{'bias':>8}")
    for thr in THRESHOLDS:
        h, m, f = pooled[thr]
        print(f"{f'{thr:g} mm':>12}{csi(h, m, f):>9.4f}"
              f"{h / max(h + m, 1):>8.3f}{f / max(h + f, 1):>8.3f}"
              f"{(h + f) / max(h + m, 1):>8.2f}")

    print("\nThis is the truth source the model is trained against, measured against an")
    print("independent instrument. Whatever disagreement shows up here is a floor on")
    print("how well any radar-trained model can score against gauges - the model")
    print("cannot be better than the data it learned from.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
