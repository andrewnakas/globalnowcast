"""Score the 2 km CONUS nowcast against MRMS radar, at benchmark thresholds.

This is the comparison that answers "how does this stack up against published work".
Everything else in verify/ scores satellite forecasts against later satellite
observations, which shares the retrieval's own errors and runs on a 28 km grid that
mechanically inflates CSI. Here the grid is ~2 km (what DGMR, NowcastNet and SEVIR
use), the truth is an independent instrument, and the thresholds are the 1/4/8 mm/h
the literature reports.

    python verify/run_radar.py --hours 6 --every 60

Read the result with the caveat in verify/BENCHMARKS.md: the forecast is derived from
infrared and microwave satellite retrieval, while the models it is being compared to
are trained on radar and predict radar. Being behind at high thresholds is expected,
not a bug.
"""
import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE))

import nowcast  # noqa: E402
import obs  # noqa: E402
import radar  # noqa: E402
from metrics import contingency, csi, fss  # noqa: E402

# Marshall-Palmer, matching pipeline/obs.py: 1/2/4/8 mm/h in dBZ.
THRESHOLDS = {1.0: 23.0, 2.0: 27.8, 4.0: 32.6, 8.0: 37.5}
KM_PER_PX = 0.02 * 111.0


def one_case(session, t0, leads_min, gap_min, grid):
    """Anchor at t0, advect, and score every lead against radar at that valid time."""
    anchor = obs.fetch_frame(session, t0, levels=(5,), latency_min=0, grid=grid)
    prior = obs.fetch_frame(session, t0 - timedelta(minutes=gap_min),
                            levels=(5,), latency_min=0, grid=grid)
    if anchor is None or prior is None:
        print(f"  {t0:%H:%MZ}: no satellite pair")
        return []
    sat, sat_mask = anchor[0], anchor[1]

    flow = nowcast.estimate_flow(prior[0], sat, gap_min, km_per_px=KM_PER_PX)
    kmh = np.percentile(np.hypot(flow[..., 0], flow[..., 1]), 99) * KM_PER_PX * 2
    if not flow.any():
        print(f"  {t0:%H:%MZ}: flow rejected")
        return []

    out = []
    for lead in leads_min:
        truth = radar.fetch(session, t0 + timedelta(minutes=lead), grid)
        if truth is None:
            continue
        rad, rad_mask, _ = truth
        mask = sat_mask & rad_mask
        fields = {
            "persistence": sat,
            "advection": nowcast.advect(sat, flow, lead / 30.0),
        }
        out.append((lead, fields, rad, mask))
    print(f"  {t0:%H:%MZ}: flow {kmh:.0f} km/h, {len(out)} lead(s) scored")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=6, help="how far back to start")
    ap.add_argument("--every", type=int, default=60, help="minutes between cases")
    ap.add_argument("--leads", default="30,60,90,120")
    ap.add_argument("--gap", type=int, default=30, help="flow pair separation, min")
    ap.add_argument("--fss-window", type=int, default=25)
    args = ap.parse_args()

    leads = [int(x) for x in args.leads.split(",")]
    grid = obs.CONUS_2KM
    print(f"{grid}, truth = MRMS radar, flow pair {args.gap} min\n")

    session = requests.Session()
    # Radar and satellite both need to exist, so stay well behind real time.
    end = datetime.now(timezone.utc) - timedelta(minutes=max(leads) + 40)
    pooled = defaultdict(lambda: [0, 0, 0, 0])
    fss_acc = defaultdict(list)
    sat_bias = []
    n = 0

    for k in range(args.hours * 60 // args.every):
        t0 = end - timedelta(minutes=k * args.every)
        case = one_case(session, t0, leads, args.gap, grid)
        if not case:
            continue
        n += 1
        for lead, fields, rad, mask in case:
            for model, field in fields.items():
                for mmhr, dbz in THRESHOLDS.items():
                    c = contingency(field, rad, dbz, mask)
                    key = (model, lead, mmhr)
                    pooled[key] = [a + b for a, b in zip(pooled[key], c)]
                    fss_acc[key].append(fss(field, rad, dbz, args.fss_window, mask))
            # How much wetter/drier is the satellite than the radar it is scored on?
            sat_bias.append((float((fields["persistence"] >= 23.0)[mask].mean()),
                             float((rad >= 23.0)[mask].mean())))

    if not n:
        print("no cases scored")
        return 1

    print(f"\n{n} case(s), CONUS, ~2.2 km grid, verified against MRMS radar")
    for mmhr in THRESHOLDS:
        print(f"\n=== CSI at {mmhr:g} mm/h ===")
        print(f"{'lead':>7}{'persistence':>14}{'advection':>12}")
        for lead in leads:
            row = "".join(f"{csi(*pooled[(m, lead, mmhr)][:3]):>{14 if m=='persistence' else 12}.4f}"
                          for m in ("persistence", "advection"))
            print(f"{'+' + str(lead) + 'm':>7}{row}")

    print(f"\n=== FSS (window {args.fss_window} px ~ {args.fss_window*2.2:.0f} km), "
          "1 mm/h ===")
    print(f"{'lead':>7}{'persistence':>14}{'advection':>12}")
    for lead in leads:
        row = "".join(f"{np.nanmean(fss_acc[(m, lead, 1.0)]):>{14 if m=='persistence' else 12}.4f}"
                      for m in ("persistence", "advection"))
        print(f"{'+' + str(lead) + 'm':>7}{row}")

    if sat_bias:
        s, r = np.mean([b[0] for b in sat_bias]), np.mean([b[1] for b in sat_bias])
        print(f"\nwet area at 1 mm/h: satellite {s:.4f} vs radar {r:.4f} "
              f"({s / r if r else float('nan'):.2f}x)")
        print("This is the retrieval's own bias against the instrument it is scored")
        print("on, and it caps how high CSI can go regardless of the motion field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
