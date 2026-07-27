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

Anchor 2026-07-27 00:30Z, GFS cycle 00Z, truth from complete (GLB-5) observations,
pooled over the observed domain (70N–60S), with the shipped crossover of 330 min.

CSI at dBZ>=20 — the threshold most often quoted for "meaningful" rain:

| lead | persistence | advection | gfs | blend |
|------|-------------|-----------|--------|--------|
| +30m | 0.5404 | 0.6191 | 0.1993 | 0.6191 |
| +90m | 0.3261 | 0.4140 | 0.1901 | 0.4145 |
| +150m | 0.2331 | 0.3137 | 0.1944 | 0.3173 |
| +210m | 0.1767 | 0.2401 | 0.1910 | 0.2515 |
| +270m | 0.1436 | 0.1946 | 0.1891 | 0.2228 |

CSI at dBZ>=5 — the threshold the map actually renders, so this is what a viewer sees:

| lead | persistence | advection | gfs | blend |
|------|-------------|-----------|--------|--------|
| +30m | 0.6056 | 0.6758 | 0.1499 | 0.6758 |
| +90m | 0.4042 | 0.4810 | 0.1433 | 0.4841 |
| +150m | 0.3147 | 0.3941 | 0.1468 | 0.4013 |
| +210m | 0.2537 | 0.3163 | 0.1425 | 0.3326 |
| +270m | 0.2134 | 0.2692 | 0.1411 | 0.2406 |

The one place the blend still trails is the final +270m frame at 5 dBZ (0.241 against
advection's 0.269). That is the tail of the handover and the cost of having any model
in the mix that far out; it buys the heavier-threshold gains above.

**Always score at more than one threshold.** An earlier crossover of 270 min looked
fine at 20 dBZ but was much worse at 5 dBZ (0.188 against advection's 0.269 at
+270m), and on the rendered map that showed up as the whole globe filling with model
drizzle over the animation. The model's deficit is worst exactly where the colour
scale starts.

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

`nowcast.BLEND_CROSSOVER_MIN` is set to 330 min, which scores best across 5/10/20/30
dBZ together — later than the 20 dBZ crossover alone would suggest, because handing
over early costs far more at light thresholds than it gains at heavy ones. Re-check
with `--sweep-crossover 270,330,420 --thresholds 5,10,20,30`.

The model's weakness is not miscalibration. Frequency-matching its threshold to the
observed wet area *lowers* its CSI (0.194 -> 0.181), and its frequency bias runs
~1.6 (it rains over roughly 1.6x the area actually observed). Damping the model's
weak returns before blending was tried across floors of 0/15/20/25 dBZ and changed
CSI by less than 0.002, so it was dropped rather than shipped as a dead knob.

## Cost

The nowcast adds roughly 15-25 s to the hourly build, measured as ~8.5 s fetching and
regridding the two observation frames, ~1 s of optical flow, ~0.1 s of advection, and
~7 s encoding the 17 PNGs. Rendering and network dominate; the actual nowcasting maths
is nearly free.

## Caveats

These are single-case numbers. The direction (extrapolation far ahead of the model
early on) is large and consistent, but the exact crossover is regime-dependent —
later for organised frontal systems, earlier for scattered convection. The regional
split already hints at this: GFS scores ~0.15 in the tropics against ~0.21 in
midlatitudes, so observations dominate for longer near the equator. Re-run across
more dates and both hemispheres before treating 270 minutes as settled.
