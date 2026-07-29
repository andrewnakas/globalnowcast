"""The CONUS layer: MRMS radar advection blended into the HRRR forecast.

Measured on 17 live cases against MRMS truth at 1 mm/h (ml/cache_live):

    lead    advection   HRRR     blend(120/45)
    +1h     0.282       0.176    -
    +2h     0.169       0.162    -
    pooled  0.125       0.149    0.170

Radar advection wins to ~+1.5 h, HRRR wins beyond +2.5 h, and the logistic blend
beats both pooled (+14% over HRRR, at 4 mm/h too). The trained radar model lost to
raw HRRR on these same cases (0.138, and 0.106 with PM), so what ships here is the
physics: real radar moved by its own motion, handed over to the freshest HRRR
cycle. The model re-enters through pipeline/radar_model.py if a full-scale retrain
beats this blend through ml/eval_live_inputs.py.

Same contract as every optional layer: NOWCAST_CONUS=0 kills it, any failure
returns None, and the global products never notice.
"""
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hrrr  # noqa: E402
import mrms  # noqa: E402
import obs  # noqa: E402

# Fitted on ml/cache_live (2 weeks, summer): flat optimum across 90-150 min,
# 120/45 is the interior point. Refit seasonally from verify/archive_conus.jsonl.
CROSSOVER_MIN = 120.0
TAU_MIN = 45.0
HOLD_MIN = 30.0
FLOW_GAP_MIN = 30.0
HORIZON_MIN = 360.0
KM_PER_PX = 0.02 * 111.0


def predict(session, now, valids, grid=None):
    """Blended radar/HRRR fields at each requested valid time.

    Returns ({valid: dBZ field}, mask, info) or None. `valids` outside
    [anchor, anchor + HORIZON_MIN] are simply not produced.
    """
    if os.environ.get("NOWCAST_CONUS", "1") == "0":
        return None
    grid = grid or obs.CONUS_2KM
    try:
        import cv2

        import nowcast

        cv2.setNumThreads(2)

        anchor = mrms.latest_anchor(session, now)
        if anchor is None:
            print("conus: no recent MRMS frame", file=sys.stderr)
            return None
        last = mrms.fetch_rate(session, anchor, grid)
        prev = mrms.fetch_rate(session, anchor - timedelta(minutes=FLOW_GAP_MIN),
                               grid)
        if last is None or prev is None:
            print("conus: no MRMS flow pair", file=sys.stderr)
            return None
        last_dbz, last_mask, _ = last
        prev_dbz, prev_mask, prev_time = prev
        gap = (anchor - prev_time).total_seconds() / 60.0
        if gap <= 0:
            return None

        fc = hrrr.fetch_forecast(session, anchor, grid)
        if fc is None:
            return None
        h, h_mask, cycle = fc
        hrrr_by_valid = {anchor + timedelta(hours=k + 1): h[k]
                         for k in range(h.shape[0])}

        flow = nowcast.estimate_flow(prev_dbz, last_dbz, gap, km_per_px=KM_PER_PX)
        mask = last_mask & prev_mask & h_mask

        out = {}
        for valid in valids:
            minutes = (valid - anchor).total_seconds() / 60.0
            if minutes < 0 or minutes > HORIZON_MIN:
                continue
            adv = nowcast.advect(last_dbz, flow, minutes / 30.0)
            model = nowcast.gfs_at(hrrr_by_valid, valid)  # rain-rate interp
            if model is None or minutes <= HOLD_MIN:
                out[valid] = adv
                continue
            w = nowcast.blend_weight(minutes, crossover=CROSSOVER_MIN,
                                     tau=TAU_MIN, hold=HOLD_MIN)
            rate = (w * obs.dbz_to_rain(adv)
                    + (1.0 - w) * obs.dbz_to_rain(np.maximum(model, obs.FILL)))
            out[valid] = np.where(mask, obs.rain_to_dbz(rate),
                                  np.maximum(model, obs.FILL)).astype(np.float32)
        if not out:
            return None
        info = {"anchor": anchor, "cycle": cycle, "source": "radar+hrrr"}
        return out, mask, info
    except Exception as e:  # noqa: BLE001 - the layer is optional, GFS is not
        print(f"conus: skipped ({e})", file=sys.stderr)
        return None


if __name__ == "__main__":  # smoke test: python pipeline/conus.py
    import time
    from datetime import datetime, timezone

    import requests

    t = time.time()
    now = datetime.now(timezone.utc)
    valids = [now.replace(minute=0, second=0, microsecond=0)
              + timedelta(minutes=15 * k) for k in range(0, 25, 4)]
    got = predict(requests.Session(), now, valids)
    if got is None:
        sys.exit("conus layer unavailable")
    fields, mask, info = got
    print(f"anchor {info['anchor']:%Y-%m-%d %H:%MZ}, HRRR {info['cycle']:%HZ}, "
          f"{len(fields)} frames, {time.time()-t:.1f}s")
    for valid, f in sorted(fields.items()):
        print(f"  {valid:%H:%M}  wet(>=23dBZ) {(f >= 23)[mask].mean():.4f}  "
              f"max {f.max():.1f} dBZ")
