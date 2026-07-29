"""Score the ONNX radar model on inputs built exactly as production will build them.

The model trained on the dynamical.org hourly archives: MRMS `precipitation_surface`
as MP dBZ, and the HRRR *analysis* as its forecast channel. The live job feeds it the
2-minute PrecipRate feed and true HRRR *forecast* hours from whatever cycle exists at
build time. This is three serving skews at once - forecast-vs-analysis channel,
feed-vs-archive radar, tile-vs-full-frame PM (plus 1 km training tiles vs the 2 km
serving grid) - and this script measures them all end to end instead of arguing about
them separately.

Gate before pipeline/radar_model.py may ship (the plan's riskiest assumption):
    model+PM beats the fetched HRRR forecast by >=10% pooled CSI at 1 mm/h,
    is ahead at every lead, and its wet-area ratio vs radar sits in 0.85-1.25.

    python ml/eval_live_inputs.py --cases 44

Fields are cached under --cache as uint8 (the training quantisation), so re-scoring
after a change to PM or the model costs seconds, not another hour of fetches.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE.parent / "verify"))

import hrrr  # noqa: E402
import mrms  # noqa: E402
import obs  # noqa: E402
from export_pm import pm_apply  # noqa: E402
from metrics import contingency, csi  # noqa: E402

U8_LO, U8_HI = -30.0, 60.0
THRESHOLDS = {"1 mm/h": 23.0, "4 mm/h": 32.6, "8 mm/h": 37.5}
LEADS = range(1, 7)
JOB_DELAY_MIN = 10  # the hourly job runs at :10 past


def to_u8(dbz):
    return np.clip((dbz - U8_LO) * (255.0 / (U8_HI - U8_LO)), 0, 255).astype(np.uint8)


def from_u8(u8):
    return u8.astype(np.float32) * ((U8_HI - U8_LO) / 255.0) + U8_LO


def run_banded(sess, x, h, band=1000, overlap=100):
    """Run the model in horizontal bands and stitch, for frames too large to fit
    one forward pass in memory (a 0.01-degree CONUS frame OOMs a full pass). The
    overlap exceeds the 3-level UNet's receptive field, and each band's interior
    is kept, so the stitch is seamless up to floating point."""
    H = x.shape[-2]
    if H <= band + 2 * overlap:
        return sess.run(["forecast"], {"radar": x[None], "hrrr": h[None]})[0][0]
    out = np.empty((h.shape[0],) + x.shape[-2:], np.float32)
    for y0 in range(0, H, band):
        lo, hi = max(0, y0 - overlap), min(H, y0 + band + overlap)
        got = sess.run(["forecast"], {"radar": x[None, :, lo:hi],
                                      "hrrr": h[None, :, lo:hi]})[0][0]
        out[:, y0:min(H, y0 + band)] = got[:, y0 - lo:y0 - lo + band]
    return out


def build_case(anchor: datetime, cache: Path, grid=None):
    """Fetch (or reload) one case: inputs as the live job sees them, plus truth."""
    path = cache / f"{anchor:%Y%m%d%H}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}

    s = requests.Session()
    hist = mrms.fetch_history(s, anchor, grid)
    if hist is None:
        return None
    x, x_mask, _ = hist
    fc = hrrr.fetch_forecast(s, anchor, grid,
                             avail=anchor + timedelta(minutes=JOB_DELAY_MIN))
    if fc is None:
        return None
    h, h_mask, cycle = fc
    truth, t_masks = [], []
    for k in LEADS:
        got = mrms.fetch_rate(s, anchor + timedelta(hours=k), grid)
        if got is None:
            return None
        truth.append(got[0])
        t_masks.append(got[1])

    case = {"x": to_u8(x), "h": to_u8(h), "y": to_u8(np.stack(truth)),
            "x_mask": x_mask, "h_mask": h_mask, "t_mask": np.stack(t_masks),
            "cycle_offset": np.int16((anchor - cycle).total_seconds() // 3600)}
    np.savez_compressed(path, **case)
    return case


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=44)
    ap.add_argument("--every", type=int, default=6, help="hours between anchors")
    ap.add_argument("--onnx", default="ml/model/nowcast.onnx")
    ap.add_argument("--pm", default="ml/model/pm_tables.npz")
    ap.add_argument("--cache", default="ml/cache_live")
    ap.add_argument("--out", default="ml/out_live_eval.json")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--from-cache", action="store_true",
                    help="re-score the anchors already fetched into --cache")
    ap.add_argument("--res", type=float, default=0.02,
                    help="serving grid resolution in degrees; 0.01 tests the "
                         "model at its native training scale (use its own cache)")
    ap.add_argument("--half", choices=["even", "odd", "all"], default="all",
                    help="subset by date order; 'odd' is the half fit_pm_full "
                         "did not fit its tables on")
    args = ap.parse_args()

    import onnxruntime as ort

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    tables = np.load(args.pm)["ref_q"]
    grid = None
    if abs(args.res - 0.02) > 1e-9:
        grid = obs.Grid.window(50.0, 24.0, -125.0, -66.0, args.res,
                               f"conus{args.res:g}")
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        anchors = sorted(datetime.strptime(p.stem, "%Y%m%d%H")
                         .replace(tzinfo=timezone.utc)
                         for p in cache.glob("*.npz"))
    else:
        # Newest anchor whose +6h truth already exists, then back at --every spacing.
        now = datetime.now(timezone.utc)
        latest = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=7)
        anchors = [latest - timedelta(hours=args.every * i) for i in range(args.cases)]
    if args.half != "all":
        keep = 0 if args.half == "even" else 1
        anchors = [a for i, a in enumerate(sorted(anchors)) if i % 2 == keep]

    models = ("model_pm", "model_raw", "hrrr", "persistence")
    counts = {m: {lab: {k: np.zeros(3) for k in LEADS} for lab in THRESHOLDS}
              for m in models}
    wet = {"model_pm": 0.0, "hrrr": 0.0, "truth": 0.0}
    used = 0

    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(build_case, a, cache, grid): a for a in anchors}
        for fut in as_completed(futs):
            anchor = futs[fut]
            try:
                case = fut.result()
            except Exception as e:  # noqa: BLE001 - one bad case must not end the run
                print(f"{anchor:%m-%d %HZ}: {e}", file=sys.stderr)
                continue
            if case is None:
                print(f"{anchor:%m-%d %HZ}: incomplete, skipped", file=sys.stderr)
                continue

            x, h = from_u8(case["x"]), from_u8(case["h"])
            truth = from_u8(case["y"])
            pred = run_banded(sess, x, h)
            in_mask = case["x_mask"] & case["h_mask"]
            pm = np.full_like(pred, U8_LO)
            for i in range(pred.shape[0]):
                pm[i][in_mask] = pm_apply(pred[i][in_mask], tables[i])

            fields = {"model_pm": pm, "model_raw": pred, "hrrr": h,
                      "persistence": np.repeat(x[-1:], len(LEADS), axis=0)}
            for i, k in enumerate(LEADS):
                m = in_mask & case["t_mask"][i]
                for lab, thr in THRESHOLDS.items():
                    for name, f in fields.items():
                        counts[name][lab][k] += contingency(f[i], truth[i], thr, m)[:3]
                wet["model_pm"] += float((pm[i][m] >= 23.0).sum())
                wet["hrrr"] += float((h[i][m] >= 23.0).sum())
                wet["truth"] += float((truth[i][m] >= 23.0).sum())
            used += 1
            print(f"{anchor:%m-%d %HZ}: ok (cycle -{int(case['cycle_offset'])}h, "
                  f"{used} scored)")

    if used < 10:
        sys.exit(f"only {used} usable cases - not enough to gate on")

    print(f"\n{used} cases, leads +1..+6h, {args.onnx}")
    pooled = {}
    for lab in THRESHOLDS:
        print(f"\nCSI at {lab}")
        print(f"{'lead':>6}" + "".join(f"{m:>12}" for m in models))
        for k in LEADS:
            row = [csi(*counts[m][lab][k]) for m in models]
            print(f"{'+' + str(k) + 'h':>6}" + "".join(f"{v:>12.4f}" for v in row))
        pool_row = {m: csi(*sum(counts[m][lab][k] for k in LEADS)) for m in models}
        pooled[lab] = pool_row
        print(f"{'all':>6}" + "".join(f"{pool_row[m]:>12.4f}" for m in models))

    ratio = wet["model_pm"] / max(wet["truth"], 1.0)
    gain = pooled["1 mm/h"]["model_pm"] / max(pooled["1 mm/h"]["hrrr"], 1e-9) - 1
    every_lead = all(csi(*counts["model_pm"]["1 mm/h"][k]) >
                     csi(*counts["hrrr"]["1 mm/h"][k]) for k in LEADS)
    print(f"\nwet-area ratio (model+PM / truth, 1 mm/h): {ratio:.3f}")
    print(f"pooled gain over HRRR at 1 mm/h: {gain:+.1%}")
    print(f"ahead of HRRR at every lead: {every_lead}")
    ok = gain >= 0.10 and every_lead and 0.85 <= ratio <= 1.25
    print(f"\nGATE: {'PASS' if ok else 'FAIL'} "
          f"(need >=+10% pooled, every lead, ratio 0.85-1.25)")

    Path(args.out).write_text(json.dumps({
        "cases": used, "wet_ratio": ratio, "pooled_gain_1mm": gain,
        "every_lead": every_lead, "gate": "pass" if ok else "fail",
        "counts": {m: {lab: {str(k): counts[m][lab][k].tolist() for k in LEADS}
                       for lab in THRESHOLDS} for m in models}}, indent=1))
    print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
