"""Pool the accumulated verification archive and refit the handover lead.

The archive (verify/archive.jsonl) is appended to by the weekly workflow, one line
per case, holding raw contingency counts per model / lead / threshold / region.
Counts rather than scores, because counts pool exactly across cases while averaging
per-case ratios is biased toward cases with little rain.

    python verify/archive.py                 # pooled summary + best crossover
    python verify/archive.py --by season     # split by season
    python verify/archive.py --by region     # split by latitude band
    python verify/archive.py --since 2026-09 # only cases from then on

Once the archive spans more than one season, a split that shows the best crossover
moving materially is the signal to make the handover season- or latitude-dependent
rather than the single global constant in pipeline/nowcast.py.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "archive.jsonl"

# Northern-hemisphere meteorological seasons; the label is about the regime mix, and
# the observed domain spans both hemispheres, so treat these as coarse bins only.
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def csi(counts):
    h, m, f = counts[0], counts[1], counts[2]
    d = h + m + f
    return h / d if d else float("nan")


def load(path: Path, since: str | None):
    if not path.exists():
        return []
    cases = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if since and c["t0"][:len(since)] < since:
            continue
        cases.append(c)
    return cases


def pool(cases, region="global", group=None):
    """{group: {(model, lead, threshold): [h, m, f]}}"""
    out = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for c in cases:
        if group == "season":
            g = SEASONS[int(c["t0"][5:7])]
        elif group == "month":
            g = c["t0"][:7]
        else:
            g = "all"
        for r in c["records"]:
            counts = r["counts"].get(region)
            if not counts:
                continue
            k = (r["model"], r["lead_min"], r["threshold"])
            out[g][k] = [a + b for a, b in zip(out[g][k], counts)]
    return out


def summarise(tag, table):
    models = sorted({k[0] for k in table})
    leads = sorted({k[1] for k in table})
    thrs = sorted({k[2] for k in table})
    if not models:
        return None

    # Mean CSI over every lead x threshold cell, plus how often each variant is at
    # least as good as pure advection - the invariant the blend has to keep.
    ranked = []
    for m in models:
        vals = [csi(table[(m, l, t)]) for l in leads for t in thrs
                if (m, l, t) in table]
        if not vals:
            continue
        wins = sum(1 for l in leads for t in thrs
                   if (m, l, t) in table and ("advection", l, t) in table
                   and csi(table[(m, l, t)]) >= csi(table[("advection", l, t)]) - 1e-9)
        cells = sum(1 for l in leads for t in thrs if (m, l, t) in table)
        ranked.append((sum(vals) / len(vals), m, wins, cells))
    ranked.sort(reverse=True)

    print(f"\n=== {tag} ===")
    print(f"{'model':>12}{'mean CSI':>11}{'>= advection':>15}")
    for mean, m, wins, cells in ranked:
        print(f"{m:>12}{mean:>11.4f}{f'{wins}/{cells}':>15}")

    # Only crossover variants are candidates for the shipped constant.
    cand = [(mean, m, wins, cells) for mean, m, wins, cells in ranked
            if m.startswith("x") and m[1:].replace(".", "").isdigit()]
    if cand:
        clean = [c for c in cand if c[2] == c[3]]  # never worse than advection
        best = (clean or cand)[0]
        note = "" if clean else "  (none beat advection everywhere)"
        print(f"  best crossover: {best[1][1:]} min{note}")
        return float(best[1][1:])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(DEFAULT))
    ap.add_argument("--by", choices=("season", "month", "region"), default=None)
    ap.add_argument("--since", default=None, help="ISO prefix, e.g. 2026-09")
    args = ap.parse_args()

    cases = load(Path(args.path), args.since)
    if not cases:
        print(f"no cases in {args.path}"
              + (f" since {args.since}" if args.since else "")
              + "\nthe weekly workflow appends to it; see verify/README.md")
        return 1

    span = f"{min(c['t0'] for c in cases)[:10]} .. {max(c['t0'] for c in cases)[:10]}"
    print(f"{len(cases)} case(s), {span}")

    if args.by == "region":
        picks = {}
        for region in ("global", "tropics", "midlat"):
            table = pool(cases, region=region)["all"]
            picks[region] = summarise(f"region: {region}", table)
        vals = [v for v in picks.values() if v is not None]
        if len(set(vals)) > 1:
            print("\nThe best crossover differs by latitude "
                  f"({picks}) - worth making the handover latitude-dependent.")
        return 0

    grouped = pool(cases, group=args.by)
    picks = {}
    for g in sorted(grouped):
        picks[g] = summarise(f"{args.by or 'pooled'}: {g}", grouped[g])
    if args.by and len({v for v in picks.values() if v is not None}) > 1:
        print(f"\nThe best crossover differs by {args.by} ({picks}).")
        print("If that holds as more cases land, make the handover conditional")
        print("rather than the single constant in pipeline/nowcast.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
