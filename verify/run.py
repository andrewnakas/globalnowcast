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
TROPICS = np.abs(obs.GFS_LAT) < 23.5
# Set via --sweep-crossover to compare candidate handover leads in one pass.
SWEEP_CROSSOVERS: tuple[float, ...] = ()


def _cached(name: str, build):
    """Fetches are slow and parameter sweeps repeat them; keep them on disk."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    data = build()
    if data is not None:
        np.savez_compressed(path, **data)
    return data


def load_obs(session, when, levels=(5,)):
    """Observation at `when`, complete frames only so the domain is stable."""
    def build():
        got = obs.fetch_frame(session, when, levels=levels, latency_min=0)
        if got is None:
            return None
        dbz, mask, valid, _ = got
        if abs((valid - when).total_seconds()) > 900:
            return None  # too far from the requested time to be that case's truth
        return {"dbz": dbz, "mask": mask}

    return _cached(f"obs_{when:%Y%m%d%H%M}", build)


def load_gfs(session, cycle, fh):
    def build():
        return {"dbz": np.maximum(decode_refc(fetch_refc(session, cycle, fh)),
                                  obs.FILL).astype(np.float32)}

    got = _cached(f"gfs_{cycle:%Y%m%d%H}_f{fh:03d}", build)
    return got["dbz"] if got else None


def run_case(session, t0, leads, gap_min=30):
    """One anchor time -> {model: {lead: field}}, plus truth and the scoring mask."""
    anchor = load_obs(session, t0)
    prior = load_obs(session, t0 - timedelta(minutes=gap_min))
    if anchor is None or prior is None:
        print(f"  {t0:%Y-%m-%d %H:%MZ}: observations unavailable, skipping")
        return None

    flow = nowcast.estimate_flow(prior["dbz"], anchor["dbz"], gap_min)
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
        gfs = load_gfs(session, cycle, fh)
        if gfs is None:
            continue

        steps = lead / 30.0
        adv = nowcast.advect(anchor["dbz"], flow, steps)
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
    args = ap.parse_args()

    global SWEEP_CROSSOVERS, MODELS
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

    hours = args.days * 24
    for offset in range(0, hours, args.every):
        t0 = start + timedelta(hours=offset)
        print(f"case {t0:%Y-%m-%d %H:%MZ}")
        case = run_case(session, t0, leads)
        if not case:
            continue
        n += 1
        for lead, valid, fields, truth, mask in case:
            for model, field in fields.items():
                for thr in thresholds:
                    for region, sel in (("global", mask),
                                        ("tropics", mask & TROPICS[:, None]),
                                        ("midlat", mask & ~TROPICS[:, None])):
                        c = contingency(field, truth, thr, sel)
                        key = (model, lead, thr, region)
                        pooled[key] = [a + b for a, b in zip(pooled[key], c)]
                    fss_acc[(model, lead, thr)].append(
                        fss(field, truth, thr, args.fss_window, mask))
                    records.append({"t0": t0.isoformat(), "lead_min": lead,
                                    "model": model, "threshold": thr,
                                    **scores(*contingency(field, truth, thr, mask))})
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

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=1))
        print(f"\nwrote {len(records)} records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
