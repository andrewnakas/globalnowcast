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

## The check that should gate it

Before this ships, run `verify/run_asos.py` on the model's output rather than on MRMS.
Radar and gauges already disagree at CSI 0.27 with a wet bias of 2.2, so the question
is whether the model inherits that bias or amplifies it. A model that beats HRRR on
radar while painting rain over three times the gauge-measured area is not obviously an
improvement for anyone reading the map.

## Honest summary of where the model stands

It beats HRRR by 22.7% at 1 mm/h and by more at heavier thresholds, on held-out months,
reproducibly across two seeds (0.5239 and 0.5228). It sits above the intrinsic
hour-to-hour predictability of the field, so it is predicting change rather than only
tracking. And it fits the CPU budget.

What it is not: comparable to any published benchmark (wrong cadence, wrong grid, wrong
truth - see `../verify/BENCHMARKS.md`), global, or validated against gauges. The first
of those is addressed by the sub-hourly dataset builder, the last by the ASOS harness,
and neither has been run to completion yet.
