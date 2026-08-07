"""Does a third model arm beat the shipped GFS+AIFS mean?

The two-model mean is live and measured (+15/+31/+19% CSI at 5/10/20 dBZ over
raw GFS). GEFS is the obvious third: same catalog, same grid, and it carries 31
ensemble members, so its member-mean is a genuinely different kind of forecast -
an ensemble average is smoother and better calibrated in area than any single
deterministic run.

Two reasons it might not pay, and both need measuring rather than assuming:

  * GEFS initialises every 24 h against GFS's 6 h, so at a 6-48 h lead it can be
    most of a day staler. Staleness costs more than ensemble smoothing gains.
  * An ensemble mean is smooth by construction. Averaging it into an already-
    averaged field risks compounding the blur that probability matching exists
    to undo, which would show up as a loss at the high thresholds.

Scored on the harvested pairs against RRQPE, the same cases and truth every
other arm decision in this project used.

    python ml/eval_three_model.py
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE.parent / "verify"))

from eval_aifs_vs_gfs import aifs_dbz, from_u8, open_aifs, rain_to_dbz  # noqa: E402
from metrics import contingency, csi  # noqa: E402

THRESHOLDS = (5.0, 10.0, 20.0, 30.0)
GEFS_BUCKET = "dynamical-noaa-gefs"
GEFS_PREFIX = "noaa-gefs-forecast-35-day/v0.2.0.icechunk"
VAR = "precipitation_surface"
# A handful of members is enough for a mean and keeps the read affordable; the
# spread across 31 members is dominated by the first few.
MEMBERS = 5


def open_gefs():
    import icechunk
    import xarray as xr

    st = icechunk.s3_storage(bucket=GEFS_BUCKET, prefix=GEFS_PREFIX,
                             region="us-west-2", anonymous=True)
    repo = icechunk.Repository.open(st)
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False,
                        chunks=None)


def gefs_dbz(ds, valid, max_age_h=30):
    """GEFS member-mean precipitation valid at `valid`, as dBZ on the GFS grid.

    Picks the newest init that is at least 6 h before the valid time, mirroring
    what an hourly job could actually have used.
    """
    inits = ds.init_time.values
    want = np.datetime64(valid.replace(tzinfo=None), "ns")
    usable = inits[inits <= want - np.timedelta64(6, "h")]
    if not len(usable):
        return None, None
    init = usable[-1]
    lead_h = int((want - init) / np.timedelta64(1, "h"))
    if lead_h > max_age_h + 48:
        return None, None
    try:
        sel = ds[VAR].sel(init_time=init,
                          lead_time=np.timedelta64(lead_h, "h"))
        sel = sel.isel(ensemble_member=slice(0, MEMBERS))
        rate = np.nan_to_num(sel.values.astype(np.float32)).mean(axis=0) * 3600.0
    except Exception as e:  # noqa: BLE001
        print(f"  gefs {valid:%m-%d %HZ}: {type(e).__name__}", file=sys.stderr)
        return None, None
    lat, lon = ds.latitude.values, ds.longitude.values
    if lat[0] < lat[-1]:
        rate = rate[::-1]
    if lon.max() > 180.0:
        rate = np.roll(rate, -int((lon >= 180.0).argmax()), axis=1)
    return rain_to_dbz(rate), lead_h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="ml/gfs_pairs")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    import obs

    aifs_ds, gefs_ds = open_aifs(), open_gefs()
    inits = aifs_ds.init_time.values
    paths = sorted(Path(args.pairs).glob("*.npz"))[-args.limit:]

    names = ("gfs", "aifs", "gefs", "two_model", "three_model")
    counts = {n: {t: np.zeros(3) for t in THRESHOLDS} for n in names}
    wet = {n: 0.0 for n in names}
    wet["obs"] = 0.0
    used = 0
    ages = []

    for p in paths:
        valid = datetime.strptime(p.stem, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        z = np.load(p)
        truth, mask = from_u8(z["obs"]), z["mask"]
        ge, age = gefs_dbz(gefs_ds, valid)
        if ge is None:
            print(f"{p.stem}: no usable GEFS", flush=True)
            continue
        scored = False
        for k in z.files:
            if not k.startswith("gfs_"):
                continue
            lead = int(k[4:])
            init = np.datetime64(valid.replace(tzinfo=None), "ns") \
                - np.timedelta64(lead, "h")
            if init not in inits:
                continue
            ai = aifs_dbz(aifs_ds, init.astype("datetime64[ns]").item(), lead)
            if ai is None:
                continue
            g = from_u8(z[k])
            if ai.shape != g.shape or ge.shape != g.shape:
                continue
            rg, ra, rge = (obs.dbz_to_rain(x) for x in (g, ai, ge))
            fields = {
                "gfs": g, "aifs": ai, "gefs": ge,
                "two_model": obs.rain_to_dbz(0.5 * (rg + ra)),
                "three_model": obs.rain_to_dbz((rg + ra + rge) / 3.0),
            }
            for n, f in fields.items():
                for t in THRESHOLDS:
                    counts[n][t] += contingency(f, truth, t, mask)[:3]
                wet[n] += float((f >= 20.0)[mask].sum())
            wet["obs"] += float((truth >= 20.0)[mask].sum())
            scored = True
        if scored:
            used += 1
            ages.append(age)
        print(f"{p.stem}: {'scored' if scored else 'skipped'} "
              f"(gefs lead {age}h)", flush=True)

    if not used:
        sys.exit("nothing scored")

    print(f"\n{used} valid times, truth = RRQPE, GEFS lead "
          f"{min(ages)}-{max(ages)}h ({MEMBERS}-member mean)")
    print(f"{'':>12}" + "".join(f"{t:>9g}" for t in THRESHOLDS)
          + f"{'wet vs obs':>12}")
    for n in names:
        print(f"{n:>12}" + "".join(f"{csi(*counts[n][t]):>9.4f}" for t in THRESHOLDS)
              + f"{wet[n] / max(wet['obs'], 1.0):>12.2f}")

    base = {t: csi(*counts["two_model"][t]) for t in THRESHOLDS}
    gains = [csi(*counts["three_model"][t]) / max(base[t], 1e-9) - 1
             for t in THRESHOLDS]
    print("\nthree_model vs the shipped two_model: "
          + "  ".join(f"{t:g}dBZ {g:+.1%}" for t, g in zip(THRESHOLDS, gains)))
    worst = min(gains)
    print("verdict: " + ("ship it" if worst > 0.0 else
                         "keep the two-model mean"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
