# Radar-target nowcast model: results

Trained on MRMS radar with HRRR as a side input, both from the dynamical.org catalog.
Verified against held-out radar, split by month so no near-duplicate hours leak between
train and validation.

## Headline

18,551 sequences, 1,236 held-out samples across 10 months, base=64:

| threshold | model | + PM | HRRR | persistence |
|-----------|-------|------|------|-------------|
| 1 mm/h | **0.5102** | 0.5028 | 0.4395 | 0.2019 |
| 2 mm/h | 0.3947 | 0.3977 | 0.3122 | 0.1152 |
| 4 mm/h | 0.2355 | **0.2946** | 0.1747 | 0.0568 |
| 8 mm/h | 0.1521 | **0.2319** | 0.1337 | 0.0262 |

By lead at 1 mm/h: 0.543 / 0.518 / 0.521 / 0.487 / 0.502 / 0.499 for +1 h to +6 h,
against HRRR's 0.446 / 0.433 / 0.437 / 0.428 / 0.456 / 0.439. Ahead at every lead, and
five times persistence by +6 h.

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
| probability matching, 8 mm/h | +0.080 | real and large |
| probability matching, 4 mm/h | +0.059 | real |
| gated skip fusion | −0.002 | inside the noise |
| wet_weight 2 vs 4 vs 8 | 8 best by +0.15 | real, and already tuned |
| base 96, gated 96, longer schedules | not run | cannot clear the spread |

Data and post-processing move the model. Architecture, at this scale and validation
size, does not.

## Probability matching does the heavy-rain work, not the network

At 8 mm/h the raw model is *worse* than HRRR at short leads - 0.111 against 0.133 at
+1 h. It finds the rain but paints it too broadly: frequency bias 1.54 with POD 0.857
against HRRR's 1.09 and 0.640.

Replacing the model's intensity distribution with the observed climatology, fitted per
lead, fixes the footprint while keeping the placement:

| lead | model | + PM | HRRR |
|------|-------|------|------|
| +1h | 0.111 | 0.225 | 0.133 |
| +3h | 0.130 | 0.237 | 0.128 |
| +6h | 0.158 | 0.203 | 0.135 |

Roughly 1.7x HRRR at every lead. **The network contributes placement; PM contributes
footprint.** Saying "the model beats HRRR by 70% on heavy rain" without that split
would be misleading.

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

**Every headline above is base=64 and cannot deploy.** They stand as an upper bound on
what this architecture and data can do; the shippable candidate is base=32 and is
being trained on the combined ~48,000-sequence dataset. Re-check with
`python ml/bench_cpu.py` before shipping anything.

## Reproducing

```bash
python ml/data/build_tiles.py --tiles 900 --out ml/tiles_big   # ~3 seq/s, network-bound
python ml/train_nowcast.py --tiles ml/tiles_big --base 32 --epochs 70 --val-months 10
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
