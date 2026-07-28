# Shipping the radar model alongside the satellite nowcast

The site currently renders two things: a global satellite nowcast (optical-flow
advection blended into GFS) and the GFS forecast behind it. The trained radar model is
a third product, not a replacement for either, and the distinction matters because
they fail in different places.

| | satellite nowcast | radar model |
|---|---|---|
| coverage | 70N-60S, ~90% of the globe | CONUS only |
| input | geostationary rain rate | MRMS radar + HRRR |
| skill vs its own truth | CSI 0.63 at +30 min | CSI 0.51 at +1 h |
| skill vs radar | **0.19** at +30 min | n/a, radar is its truth |
| lead times | 0-4 h at 15 min | +1 h to +6 h, hourly |

The satellite product's own number looks better and means less: measured against radar
rather than against itself it drops to 0.19, because the retrieval agrees with radar at
only CSI 0.18 before any forecasting happens (see `../verify/BENCHMARKS.md`). The radar
model has no such gap - its input *is* the truth source.

So over CONUS the radar model should win outright, and everywhere else the satellite
product is the only thing that runs at all. That is the split worth shipping, rather
than trying to pick one.

## What is ready

- `ml/export_nowcast.py` produces a single 3.8 MB ONNX file, verified against torch to
  5e-05 dBZ at shapes the exporter never saw, including 521x1440 global and 1301x2951
  CONUS 2 km.
- `ml/bench_cpu.py` confirms base=32 runs at 0.75 s per frame on two cores, inside the
  hourly job's ~1 s budget. base=48 and above do not.
- `pipeline/correct.py` already shows the pattern: load ONNX through onnxruntime, no
  torch in `requirements.txt`, and no-op cleanly when the model file is absent.

## What is not

The model needs its two inputs at inference time, and neither is currently fetched by
the hourly job:

- **MRMS radar** - `verify/radar.py` reads the 2-minute feed already, but the model was
  trained on the hourly dynamical.org product. The two are the same instrument at
  different cadences, so the hourly job would need to build an equivalent input from
  the raw feed rather than reuse the training path.
- **HRRR forecast** - `ml/data/dynamical.py` reads the analysis archive. The live job
  needs the *forecast*, from a different collection.

Both are anonymous S3, so this is plumbing rather than a permissions problem, but it is
real work and it is where the remaining risk sits.

## The bias gate: probability matching is not optional

`ml/check_bias.py` answers it. Wet-area fraction on held-out tiles:

| threshold | radar | HRRR | model | model + PM | model/radar |
|-----------|-------|------|-------|------------|-------------|
| 0.2 mm | 0.3272 | 0.3218 | 0.4427 | 0.3169 | 1.35 |
| 1 mm | 0.2141 | 0.2113 | 0.3274 | 0.2071 | 1.53 |
| 4 mm | 0.0440 | 0.0517 | 0.0382 | 0.0446 | 0.87 |
| 8 mm | 0.0174 | 0.0238 | 0.0071 | 0.0192 | **0.41** |

The raw model paints 1.53x the radar's wet area at 1 mm/h. Radar already runs ~2.2x the
gauge-measured area, so the raw model would sit near **3.4x against gauges** - the
failure this gate was written to catch.

With probability matching it lands at 2.1x, which is the instrument's bias rather than
the model's: the PM column tracks the radar column to within a few percent at every
threshold.

The 8 mm/h row is the one CSI never showed. The raw model produces **41% of the heavy
rain that is actually there** - it hedges toward light rain everywhere, which the
weighted loss encourages and which a threshold-averaged score hides. PM corrects both
directions at once, because it replaces the whole distribution rather than shifting a
threshold.

So probability matching is a shipping requirement, not a post-hoc improvement. The raw
model would simultaneously over-warn for drizzle and under-warn for the heavy rain that
matters.

## Honest summary of where the model stands

It beats HRRR by 22.7% at 1 mm/h and by more at heavier thresholds, on held-out months,
reproducibly across two seeds (0.5239 and 0.5228). It sits above the intrinsic
hour-to-hour predictability of the field, so it is predicting change rather than only
tracking. And it fits the CPU budget.

What it is not: comparable to any published benchmark (wrong cadence, wrong grid, wrong
truth - see `../verify/BENCHMARKS.md`), global, or validated against gauges. The first
of those is addressed by the sub-hourly dataset builder, the last by the ASOS harness,
and neither has been run to completion yet.
