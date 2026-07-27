# Verification

Scores the shipped nowcast against later observations. Nothing here reimplements
pipeline logic — it imports `pipeline/` directly, so what is measured is what runs.

```bash
python verify/test_metrics.py                        # unit checks, no network
python verify/run.py --start 2026-07-27 --days 1 --every 24 --thresholds 20
```

Fetched fields are cached under `verify/cache/` (gitignored), so re-runs and
parameter sweeps are fast.

## Baseline

14 cases twice daily over 2026-07-20..27, truth from complete (GLB-5) observations,
each scored against the GFS cycle actually available at the time, pooled over the
observed domain (70N-60S), with the shipped crossover of 360 min. Reproduce with:

```bash
python verify/run.py --start 2026-07-20 --days 7 --every 12 \
                     --leads 30,90,150,210,270 --thresholds 5,10,20,30
```

CSI at dBZ>=20 — the threshold most often quoted for "meaningful" rain:

| lead | persistence | advection | gfs | blend |
|------|-------------|-----------|--------|--------|
| +30m | 0.5547 | 0.6477 | 0.2191 | 0.6477 |
| +90m | 0.3508 | 0.4394 | 0.2148 | 0.4398 |
| +150m | 0.2691 | 0.3428 | 0.2161 | 0.3455 |
| +210m | 0.2161 | 0.2753 | 0.2155 | 0.2847 |
| +270m | 0.1804 | 0.2249 | 0.2107 | 0.2509 |

CSI at dBZ>=5 — the threshold the map actually renders, so this is what a viewer sees:

| lead | persistence | advection | gfs | blend |
|------|-------------|-----------|--------|--------|
| +30m | 0.6219 | 0.6985 | 0.1705 | 0.6985 |
| +90m | 0.4379 | 0.5201 | 0.1667 | 0.5217 |
| +150m | 0.3549 | 0.4302 | 0.1675 | 0.4373 |
| +210m | 0.2987 | 0.3625 | 0.1667 | 0.3774 |
| +270m | 0.2583 | 0.3099 | 0.1631 | 0.3182 |

**Always score at more than one threshold.** An earlier crossover of 270 min looked
fine at 20 dBZ but was much worse at 5 dBZ, and on the rendered map that showed up as
the whole globe filling with model drizzle over the animation. The model's deficit is
worst exactly where the colour scale starts. Averaged over 5/10/20/30 dBZ and all
leads, the handover candidates rank:

| crossover | 5 dBZ | 10 dBZ | 20 dBZ | 30 dBZ | mean | never worse than advection |
|-----------|-------|--------|--------|--------|------|------|
| advection | 0.4642 | 0.4468 | 0.3860 | 0.2668 | 0.3909 | — |
| 270 min | 0.4322 | 0.4426 | 0.3980 | 0.2721 | 0.3862 | 14/20 |
| 330 min | 0.4611 | 0.4565 | 0.3962 | 0.2702 | 0.3960 | 19/20 |
| **360 min** | 0.4706 | 0.4576 | 0.3937 | 0.2691 | **0.3978** | **20/20** |
| 420 min | 0.4742 | 0.4547 | 0.3891 | 0.2676 | 0.3964 | 20/20 |
| 480 min | 0.4700 | 0.4499 | 0.3870 | 0.2671 | 0.3935 | 20/20 |

360 is a real interior optimum, and the shortest handover that never loses to pure
advection anywhere. Shorter crossovers score better at 20-30 dBZ but pay for it at
the light thresholds that dominate what the map looks like.

Two invariants any change must preserve:

- **advection > persistence** at every lead — otherwise the motion field is broken.
  The most likely cause is the uint8 conversion in `nowcast.to_u8`: Farneback on
  float input silently returns zero flow, which degrades advection to persistence.
  `run.py` warns when the 99th-percentile flow is under 1 px for this reason.
- **blend >= max(advection, gfs)** at every lead — the blend must never be worse
  than either thing it is made of.

## What the numbers say

GFS sits near 0.19 at 20 dBZ regardless of lead, while extrapolation starts three
times higher and only decays past it around +4.5 h. At 5 dBZ the model is weaker
still (~0.14) and extrapolation stays ahead through the whole 4 h product.

`nowcast.BLEND_CROSSOVER_MIN` is set to 360 min, which scores best across 5/10/20/30
dBZ together — later than the 20 dBZ crossover alone would suggest, because handing
over early costs far more at light thresholds than it gains at heavy ones. Re-check
with `--sweep-crossover 300,360,420 --thresholds 5,10,20,30`.

The model's weakness is not miscalibration. Frequency-matching its threshold to the
observed wet area *lowers* its CSI, and its frequency bias runs ~1.6 (it rains over
roughly 1.6x the area actually observed). Damping the model's weak returns before
blending was tried across floors of 0/15/20/25 dBZ and changed CSI by less than
0.002, so it was dropped rather than shipped as a dead knob.

`python verify/gfs_check.py` re-tests that on everything in the cache, because it is
the sole reason the blend does not bias-correct the model first. Across 45 cached
cases frequency matching hurt **every single time**, mean -0.0078 CSI. The model
misplaces rain rather than merely over-producing it, so no rescaling can fix it.

The same script splits GFS's score by latitude. Its CSI is flat across lead time
*within every band*, so the flatness is real rather than an artifact of easy-to-hit
wet regions. Note CSI does not track how wet a band is: the tropics are the driest
band scored (wet fraction 0.029) yet score the lowest (0.163), while the wettest
band (0.066) sits mid-table.

| band | wet frac | CSI | freq bias |
|------|----------|-----|-----------|
| tropics 0-23 | 0.0290 | 0.1626 | 1.65 |
| subtropics 23-40 | 0.0344 | 0.2640 | 1.34 |
| midlatitudes 40-60 | 0.0399 | 0.2242 | 1.62 |
| high 60-70 | 0.0662 | 0.2299 | 1.01 |

The tropics are genuinely where the model is weakest — convection there is
small-scale and not resolved at 0.25 degrees — and that is also where extrapolation
stays ahead of it longest.

## Cost

The nowcast adds roughly 15-25 s to the hourly build, measured as ~8.5 s fetching and
regridding the two observation frames, ~1 s of optical flow, ~0.1 s of advection, and
~7 s encoding the 17 PNGs. Rendering and network dominate; the actual nowcasting maths
is nearly free.

## The accumulating archive

The crossover was fitted on summer cases, and it is regime-dependent — later for
organised frontal systems, earlier for scattered convection — so it may well move in
winter. That refit cannot be done by backfilling, because the GFS archive on S3 only
retains a couple of weeks. Cases have to be collected as they happen.

`.github/workflows/verify.yml` runs weekly, scores a few recent days, and appends the
contingency counts to `verify/archive.jsonl` (one line per case, ~23 KB, counts only
— no fields). It is deliberately separate from the hourly site build and cannot
delay or fail a deploy; `verify/**` is not in that workflow's path filter, so the
archive commits do not trigger a rebuild. It scores with `NOWCAST_CORRECT=0` so rows
stay comparable as the ML model comes and goes.

Refit at any time from whatever has accumulated:

```bash
python verify/archive.py                  # pooled, and the best crossover
python verify/archive.py --by season      # has the answer moved between seasons?
python verify/archive.py --by region      # tropics vs midlatitudes
python verify/archive.py --since 2026-12  # only recent cases
```

It ranks every candidate by mean CSI across all lead/threshold cells and reports how
often each is at least as good as pure advection — the invariant the blend must keep.
When a split shows the best crossover moving materially, that is the signal to make
the handover conditional rather than the single constant in `pipeline/nowcast.py`.

At 18 cases the answer is still 360 min. `--by region` already hints at the first
refinement worth making: the tropics prefer a later handover (420) than the
midlatitudes (360), which matches GFS being weakest there. That is one season of
data, so it is noted rather than acted on.

## Caveats

All cases so far are late July 2026 — one season. The archive above exists to fix
that, but it needs months of wall-clock time before a seasonal split means anything.
Until then, treat the handover as fitted for summer.

Skill also varies strongly by latitude. GFS scores ~0.16 CSI in the tropics against
~0.26 in the subtropics, so extrapolation stays ahead of it for longer near the
equator. A single global handover is a compromise; a latitude-dependent crossover is
the obvious next refinement, but it should not be added until more than one season
supports it.
