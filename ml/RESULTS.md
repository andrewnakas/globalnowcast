# Radar-target nowcast model: results

Trained on MRMS radar with HRRR as a side input, both from the dynamical.org catalog.
Verified against held-out radar, split by month so no near-duplicate hours leak between
train and validation.

## Headline

48,316 sequences, base=32 - the largest size that fits the hourly job. 1,500 held-out
samples, months held out so no near-duplicate hours leak between train and validation:

| threshold | model | + PM | HRRR | persistence |
|-----------|-------|------|------|-------------|
| 1 mm/h | **0.5055** | 0.5031 | 0.4125 | 0.2143 |
| 2 mm/h | **0.4169** | 0.4146 | 0.3164 | 0.1441 |
| 4 mm/h | 0.3255 | **0.3378** | 0.2388 | 0.0995 |
| 8 mm/h | 0.1790 | **0.2421** | 0.1542 | 0.0615 |

That is +22.5% / +31.8% / +36.3% / +16.1% over HRRR raw, and +22.0% / +31.0% / +41.5%
/ +57.0% with probability matching. By lead at 1 mm/h the model runs +28.5% at +1 h
easing to +18.0% at +6 h, ahead of HRRR everywhere and about four times persistence by
+6 h.

The shippable size is also the best size. An earlier base=64 run peaked at 0.5102 on a
smaller dataset; base=32 on the full set reaches 0.5239 with a quarter of the
parameters, and unlike base=64 it runs inside the hourly job's CPU budget. More data at
a deployable size beat more parameters at an undeployable one.

## How much headroom is actually left

Consecutive *truth* frames agree with each other at CSI 0.39-0.42 at 1 mm/h - that is
how much the rain field genuinely changes in one hour. A forecast that simply carried
the observations forward perfectly would score about 0.40.

The model scores 0.51, so it is already above that line: it is not merely tracking, it
is predicting some of the change. Which also means the remaining headroom is not the
distance to 1.0. Chasing a leaderboard-style number past this point runs into the
atmosphere being genuinely stochastic at these scales, not into a model limitation.

Two consequences worth being concrete about. Gains from here will be small and hard to
measure - the run-to-run spread on this dataset is 0.003, so a real improvement now
looks like +0.01, not +0.05. And a claim of "beating the benchmarks" needs to say which
benchmark, on whose data, at what resolution: see
[../verify/BENCHMARKS.md](../verify/BENCHMARKS.md), where the honest answer is that
nobody's leaderboard scores this task on this grid against this truth.

## Read this before quoting a number

**Every figure carries about ±0.016.** Two runs of an identical configuration - same
data, same hyperparameters, different seed - peaked 0.5102 and 0.4946. Across 39
matched epochs their standard deviation was 0.0109 and they differed by up to 0.038 at
a single epoch.

So the gain over HRRR (+0.07 at 1 mm/h) is about four times the run-to-run spread and
is real. Differences between architecture variants (all under 0.008) are not
measurable at this validation size, and were mistaken for results until the spread was
measured.

## What moved the number, and what didn't

| change | effect | verdict |
|--------|--------|---------|
| 5x more data (3.3k → 18.5k) | +0.039 | real, ~2.5x the spread |
| 2.6x more again (18.5k → 48k) | +0.013 | real but smaller; returns diminishing |
| probability matching, 8 mm/h | +0.080 | real and large |
| probability matching, 4 mm/h | +0.059 | real |
| gated skip fusion | −0.002 | inside the noise |
| wet_weight 2 vs 4 vs 8 | 8 best by +0.15 | real, and already tuned |
| base 96, gated 96, longer schedules | not run | cannot clear the spread |

Data and post-processing move the model. Architecture, at this scale and validation
size, does not.

## Probability matching does the heavy-rain work, not the network

The model finds far more rain than HRRR but paints it too broadly - frequency bias 1.53
with POD 0.849, against HRRR's 0.99 and 0.580. Replacing its intensity distribution
with the observed climatology, fitted per lead, fixes the footprint while keeping the
placement, and at 8 mm/h that is worth more than everything the network does:

| threshold | model | + PM | HRRR |
|-----------|-------|------|------|
| 1 mm/h | 0.5055 | 0.5031 | 0.4125 |
| 4 mm/h | 0.3255 | **0.3378** | 0.2388 |
| 8 mm/h | 0.1790 | **0.2421** | 0.1542 |

PM adds 35% at 8 mm/h and 4% at 4 mm/h, for about half a percent given up at 1 mm/h.
**The network contributes placement; PM contributes footprint.** Quoting the combined
+57% at 8 mm/h as a property of the model would be misleading - most of it is the
distribution correction.

On the smaller datasets PM was a straight trade, buying the heavy end at the cost of
the light. With 48k sequences it is close to free, and the raw model no longer loses
to HRRR at 8 mm/h before correction the way it used to.

Fit per lead, not pooled: the pooled fit dips to 0.4518 at +3 h at 1 mm/h where
per-lead holds 0.4993, because one distribution across six hours is wrong for the
early and late leads in opposite directions.

## What can actually ship

The hourly job runs on GitHub Actions, about two cores, roughly a second per rendered
frame. Measured on a 521x1440 global grid:

| base | params | s/frame | |
|------|--------|---------|---|
| 16 | 0.23M | 0.39 | ok |
| 32 | 0.93M | 0.75 | ok, the ceiling |
| 48 | 2.09M | 1.66 | too slow |
| 64 | 3.71M | 2.39 | too slow |

The headline model is base=32 and fits, at 0.75 s per frame. This was worth measuring
early and was not: an entire night of tuning went into base=64 before anyone checked
whether it could run in the job it was meant for. It could not. Re-check with
`python ml/bench_cpu.py` before shipping anything.

## Reproducing

```bash
python ml/data/build_tiles.py --tiles 900 --out ml/tiles_big   # ~3 seq/s, network-bound
python ml/train_nowcast.py --tiles ml/tiles_all --base 32 --epochs 70 --val-months 12 \
    --resume     # checkpoints every epoch to <out>.last; --resume continues from it
python ml/eval_nowcast.py --ckpt ml/model/nowcast.pt           # per lead and threshold
python ml/eval_pm_lead.py --ckpt ml/model/nowcast.pt           # with probability matching
python ml/bench_cpu.py                                         # will it run in the hourly job
```

## Limits

CONUS only - MRMS and HRRR both stop at the border, so this does not replace the global
satellite product, it sits alongside it where radar exists.

Hourly steps, because that is the cadence of the dynamical.org MRMS archive. A true
0-2 h nowcast wants the 2-minute native feed that `verify/radar.py` already reads.

Validation is 10 months sampled across 2014-2026, which spans seasons but is one
region and one instrument. The comparison against published models in
[../verify/BENCHMARKS.md](../verify/BENCHMARKS.md) still applies: this is not scored on
anyone's leaderboard, and the resolution and truth source differ from theirs.
