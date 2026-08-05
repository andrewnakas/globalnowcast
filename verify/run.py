"""Score the nowcast against observations, over as many cases as you care to run.

    python verify/run.py --start 2026-07-26 --days 1 --every 6

Compares, at each lead, on one identical mask:

    persistence   the latest observation, held fixed
    advection     that observation transported along its own motion
    gfs           the model forecast valid at the same time
    blend         what the site actually ships

Truth is a later observation. Every model runs through the production modules in
pipeline/ - nothing here reimplements the thing it is supposed to be checking.

Notes on fairness, both of which flatter GFS if you get them wrong:
  - the GFS cycle is the one that would really have been available at the anchor
    time, not a fresher one published later;
  - truth uses only fully-complete observation frames, so the verified domain does
    not silently change from lead to lead.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import correct  # noqa: E402
import nowcast  # noqa: E402
import obs  # noqa: E402
from gfs import fetch_refc, find_latest_cycle  # noqa: E402
from metrics import contingency, csi, fss, scores  # noqa: E402
from render import decode_refc  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
MODELS = ("persistence", "advection", "gfs", "blend")
# TROPICS is defined below, from GRID - a regional mask hard-coded to one
# latitude axis breaks the moment the verified grid changes.
# Set via --sweep-crossover to compare candidate handover leads in one pass.
SWEEP_CROSSOVERS: tuple[float, ...] = ()
# Set by --with-aifs: score the multi-model arm the site actually ships past the
# blend window, rather than raw GFS.
WITH_AIFS = False
# Mirrors main.NOWCAST_GFS_HOURS: production keeps the frames the satellite
# blend consumes on raw GFS and averages AIFS in only past that window. Defined
# here rather than imported so verification does not drag in the build module.
AIFS_FROM_MIN = 6 * 60


def _cached(name: str, build, attempts: int = 3):
    """Fetches are slow and parameter sweeps repeat them; keep them on disk.

    Retries transient S3 failures. A long sweep makes hundreds of requests, so a
    single dropped connection is close to certain over a full run and must not
    throw away everything scored so far.
    """
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    for attempt in range(attempts):
        try:
            data = build()
            break
        except Exception as e:  # noqa: BLE001 - retry anything the network throws
            if attempt == attempts - 1:
                print(f"  {name}: giving up after {attempts} tries ({e})")
                return None
            time.sleep(2 * (attempt + 1))
    if data is not None:
        np.savez_compressed(path, **data)
    return data


# The nowcast product renders at 0.1 degrees (pipeline/main.build_nowcast), so
# that is the grid its handover has to be fitted on: at 0.25 the same motion is
# sub-pixel for longer, which flatters advection's decay and moves the optimum.
# Cache keys carry the grid name so 0.25 and 0.1 cases never mix.
GRID = obs.GLOBAL_HI
KM_PER_PX = 0.1 * 111.0
TROPICS = np.abs(GRID.lat) < 23.5


def load_obs(session, when, levels=(5,)):
    """Observation at `when`, complete frames only so the domain is stable."""
    def build():
        got = obs.fetch_frame(session, when, levels=levels, latency_min=0,
                              grid=GRID)
        if got is None:
            return None
        dbz, mask, valid, _ = got
        if abs((valid - when).total_seconds()) > 900:
            return None  # too far from the requested time to be that case's truth
        return {"dbz": dbz, "mask": mask}

    return _cached(f"obs_{GRID.name}_{when:%Y%m%d%H%M}", build)


def load_gfs(session, cycle, fh, valid=None, with_aifs=False):
    """The shipped model arm on the verification grid.

    GFS is native 0.25; production upsamples it in rain-rate space where the
    blend consumes it (never in dBZ, which is logarithmic), so this does the
    same rather than comparing against a differently-resampled field.

    With `with_aifs`, this returns the multi-model mean the site ships past the
    blend window (9ab9e9a) rather than raw GFS - otherwise a crossover fitted
    here would be fitted against an arm production no longer uses.
    """
    def build():
        import cv2

        dbz = np.maximum(decode_refc(fetch_refc(session, cycle, fh)),
                         obs.FILL).astype(np.float32)
        if with_aifs and valid is not None:
            try:
                import aifs as aifs_mod

                # `now` must be the GFS cycle time, not the valid time: asking
                # for the newest init *at* the valid time would hand a +48h
                # forecast an init from that same moment, scoring AIFS as a
                # near-analysis and flattering it enormously.
                got = aifs_mod.fetch(cycle, [valid])
                if got and valid in got[0]:
                    dbz = obs.rain_to_dbz(
                        0.5 * (obs.dbz_to_rain(dbz)
                               + obs.dbz_to_rain(got[0][valid])))
            except Exception as e:  # noqa: BLE001 - fall back to raw GFS
                print(f"  aifs {valid:%m-%d %HZ} f{fh:03d}: {e}", file=sys.stderr)
        if dbz.shape != GRID.shape:
            rate = cv2.resize(obs.dbz_to_rain(dbz),
                              (GRID.shape[1], GRID.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            dbz = obs.rain_to_dbz(rate)
        return {"dbz": dbz}

    # The arm goes in the cache key: a GFS-only field must never silently
    # satisfy a request for the multi-model mean.
    arm = "gfsaifs" if with_aifs else "gfs"
    got = _cached(f"{arm}_{GRID.name}_{cycle:%Y%m%d%H}_f{fh:03d}", build)
    return got["dbz"] if got else None


def run_case(session, t0, leads, gap_min=30):
    """One anchor time -> {model: {lead: field}}, plus truth and the scoring mask."""
    anchor = load_obs(session, t0)
    prior = load_obs(session, t0 - timedelta(minutes=gap_min))
    if anchor is None or prior is None:
        print(f"  {t0:%Y-%m-%d %H:%MZ}: observations unavailable, skipping")
        return None

    flow = nowcast.estimate_flow(prior["dbz"], anchor["dbz"], gap_min,
                                 km_per_px=KM_PER_PX)
    p99 = float(np.percentile(np.hypot(flow[..., 0], flow[..., 1]), 99))
    if p99 < 1.0:
        # Near-zero motion means the uint8 conversion regressed and "advection"
        # has quietly become persistence. Worth shouting about.
        print(f"  {t0:%Y-%m-%d %H:%MZ}: WARNING flow p99 {p99:.2f}px - suspiciously still")

    cycle = find_latest_cycle(session, t0, horizon=max(leads) // 60 + 6)
    out = []
    for lead in leads:
        valid = t0 + timedelta(minutes=lead)
        truth = load_obs(session, valid)
        if truth is None:
            continue

        fh = int(round((valid - cycle).total_seconds() / 3600))
        # Production blends AIFS only past the satellite window (main.py keeps
        # the blend's own frames on raw GFS), so the arm scored here has to
        # follow the same rule or the fit is against a field that never ships.
        use_aifs = WITH_AIFS and lead > AIFS_FROM_MIN
        gfs = load_gfs(session, cycle, fh, valid=valid, with_aifs=use_aifs)
        if gfs is None:
            continue

        steps = lead / 30.0
        # Explicit wrap: the default keys off the classic 0.25 grid's width, so
        # a 0.1 global grid would silently grow a dry seam at the antimeridian.
        adv = nowcast.advect(anchor["dbz"], flow, steps, wrap=True)
        mask = anchor["mask"] & truth["mask"]
        fields = {
            "persistence": anchor["dbz"],
            "advection": adv,
            "gfs": gfs,
            "blend": nowcast.blend(adv, gfs, mask, lead),
        }
        for cross in SWEEP_CROSSOVERS:
            saved = nowcast.BLEND_CROSSOVER_MIN
            nowcast.BLEND_CROSSOVER_MIN = cross
            fields[f"x{cross:g}"] = nowcast.blend(adv, gfs, mask, lead)
            nowcast.BLEND_CROSSOVER_MIN = saved
        out.append((lead, valid, fields, truth["dbz"], mask))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="UTC date, YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--every", type=int, default=6, help="hours between cases")
    ap.add_argument("--hour", type=int, default=0, help="first case hour, UTC")
    ap.add_argument("--leads", default="30,90,150,210,270", help="minutes")
    ap.add_argument("--thresholds", default="10,20,30")
    ap.add_argument("--fss-window", type=int, default=25)
    ap.add_argument("--out", default=None, help="write per-case JSON here")
    ap.add_argument("--sweep-crossover", default=None,
                    help="also score these BLEND_CROSSOVER_MIN values, e.g. 180,270,360")
    ap.add_argument("--with-aifs", action="store_true",
                    help="use the shipped GFS+AIFS mean as the model arm, not "
                         "raw GFS; required for any crossover refit since the "
                         "handover is fitted against whatever arm ships")
    ap.add_argument("--archive", default=None, metavar="PATH",
                    help="append compact per-region counts to this JSONL archive, "
                         "skipping cases already in it (see verify/archive.py)")
    args = ap.parse_args()

    global SWEEP_CROSSOVERS, MODELS, WITH_AIFS
    WITH_AIFS = args.with_aifs
    if args.sweep_crossover:
        SWEEP_CROSSOVERS = tuple(float(x) for x in args.sweep_crossover.split(","))
        MODELS = MODELS + tuple(f"x{c:g}" for c in SWEEP_CROSSOVERS)

    leads = [int(x) for x in args.leads.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(
        hour=args.hour, tzinfo=timezone.utc)

    print(f"ML correction active: {correct.is_active()}"
          "  (false => 'gfs' is the raw model)")

    session = requests.Session()
    # Pooled counts, plus FSS which has to be averaged since it is already a ratio.
    pooled = defaultdict(lambda: [0, 0, 0, 0])
    fss_acc = defaultdict(list)
    records = []
    n = 0

    done = set()
    if args.archive and Path(args.archive).exists():
        for line in Path(args.archive).read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["t0"])
        print(f"archive holds {len(done)} case(s) already")

    hours = args.days * 24
    for offset in range(0, hours, args.every):
        t0 = start + timedelta(hours=offset)
        if t0.isoformat() in done:
            print(f"case {t0:%Y-%m-%d %H:%MZ}: already archived, skipping")
            continue
        print(f"case {t0:%Y-%m-%d %H:%MZ}")
        try:
            case = run_case(session, t0, leads)
        except Exception as e:  # noqa: BLE001 - one bad case must not lose the rest
            print(f"  failed, skipping: {e}")
            continue
        if not case:
            continue
        n += 1
        for lead, valid, fields, truth, mask in case:
            for model, field in fields.items():
                for thr in thresholds:
                    per_region = {}
                    for region, sel in (("global", mask),
                                        ("tropics", mask & TROPICS[:, None]),
                                        ("midlat", mask & ~TROPICS[:, None])):
                        c = contingency(field, truth, thr, sel)
                        per_region[region] = c
                        key = (model, lead, thr, region)
                        pooled[key] = [a + b for a, b in zip(pooled[key], c)]
                    fss_acc[(model, lead, thr)].append(
                        fss(field, truth, thr, args.fss_window, mask))
                    rec = {"t0": t0.isoformat(), "lead_min": lead,
                           "model": model, "threshold": thr,
                           **scores(*per_region["global"])}
                    if args.archive:
                        # Counts pool exactly across cases, so the archive keeps
                        # those rather than derived ratios - and keeps them per
                        # region so a latitude-dependent refit stays possible.
                        rec = {"t0": t0.isoformat(), "lead_min": lead,
                               "model": model, "threshold": thr,
                               "counts": {r: v[:3] for r, v in per_region.items()}}
                    records.append(rec)
            print(f"  +{lead:3d}m  " + "  ".join(
                f"{m}={csi(*contingency(fields[m], truth, 20.0, mask)[:3]):.4f}"
                for m in MODELS))

    if not n:
        print("no cases scored")
        return 1

    for thr in thresholds:
        print(f"\n=== pooled CSI, dBZ>={thr:.0f}, {n} case(s) ===")
        print("lead    " + "".join(f"{m:>13}" for m in MODELS))
        for lead in leads:
            row = "".join(f"{csi(*pooled[(m, lead, thr, 'global')][:3]):>13.4f}"
                          for m in MODELS)
            print(f"+{lead:3d}m {row}")

    print(f"\n=== pooled CSI by region, dBZ>=20 ===")
    for region in ("tropics", "midlat"):
        print(f"  {region}")
        for lead in leads:
            row = "".join(f"{csi(*pooled[(m, lead, 20.0, region)][:3]):>13.4f}"
                          for m in MODELS)
            print(f"  +{lead:3d}m {row}")

    print(f"\n=== FSS (window {args.fss_window}), dBZ>=20 ===")
    print("lead    " + "".join(f"{m:>13}" for m in MODELS))
    for lead in leads:
        row = "".join(f"{np.nanmean(fss_acc[(m, lead, 20.0)]):>13.4f}" for m in MODELS)
        print(f"+{lead:3d}m {row}")

    print(f"\n=== frequency bias, dBZ>=20 (>1 = too wet) ===")
    for m in MODELS:
        b = [pooled[(m, lead, 20.0, "global")] for lead in leads]
        vals = [(h + f) / (h + mi) if (h + mi) else float("nan")
                for h, mi, f, _ in b]
        print(f"  {m:>12}: " + "  ".join(f"{v:.2f}" for v in vals))

    if SWEEP_CROSSOVERS:
        # Rank swept crossovers across EVERY lead x threshold cell, not by
        # eyeballing one threshold. The optimum genuinely reverses with
        # intensity - shorter handovers win at dBZ>=20 while longer ones win at
        # dBZ>=5, which is the threshold the map renders - so a candidate picked
        # off the heavy-rain table alone will visibly flood the light-rain field
        # with model drizzle. Mean across cells plus a worst-cell column is the
        # comparison that survives that.
        print("\n=== crossover ranking (all lead x threshold cells) ===")
        print(f"{'setting':>10}{'mean CSI':>10}{'worst vs shipped':>18}"
              f"{'cells >= advection':>20}")
        cells = [(lead, thr) for lead in leads for thr in thresholds]
        ranked = []
        for m in [f"x{c:g}" for c in SWEEP_CROSSOVERS] + ["blend"]:
            vals, worst, at_least_adv = [], float("inf"), 0
            for lead, thr in cells:
                v = csi(*pooled[(m, lead, thr, "global")][:3])
                base = csi(*pooled[("blend", lead, thr, "global")][:3])
                adv = csi(*pooled[("advection", lead, thr, "global")][:3])
                vals.append(v)
                worst = min(worst, v - base)
                at_least_adv += v >= adv - 1e-9
            ranked.append((float(np.mean(vals)), m, worst, at_least_adv))
        for mean, m, worst, adv_ok in sorted(ranked, reverse=True):
            label = f"{m} (shipped)" if m == "blend" else m
            print(f"{label:>10}{mean:>10.4f}{worst:>18.4f}"
                  f"{adv_ok:>14}/{len(cells)}")
        print("A candidate earns a change only by winning the mean AND never "
              "losing a cell by more than the ~0.003 replicate noise.")

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=1))
        print(f"\nwrote {len(records)} records to {args.out}")

    if args.archive:
        # One line per case keeps appends atomic and the file readable in a diff.
        by_case = defaultdict(list)
        for r in records:
            by_case[r["t0"]].append({k: r[k] for k in
                                     ("lead_min", "model", "threshold", "counts")})
        path = Path(args.archive)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            for t0 in sorted(by_case):
                fh.write(json.dumps({"t0": t0, "records": by_case[t0]},
                                    separators=(",", ":")) + "\n")
        print(f"\nappended {len(by_case)} case(s) to {args.archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
