# Training a radar-target precipitation model

The satellite nowcast is at its ceiling. Scored against MRMS radar at zero lead — no
forecast at all — the retrieval agrees with radar at only CSI 0.18 (POD 0.22, FAR
0.50); the +30 min forecast scores 0.186. The motion model is not the limitation, the
sensor is, and no amount of training changes that while the input is a satellite
retrieval predicting itself. See [../verify/BENCHMARKS.md](../verify/BENCHMARKS.md).

Training against **radar targets** is the one move that can raise that ceiling: a model
can learn the systematic part of the retrieval's disagreement with radar. It cannot
learn what the sensor never saw, so expect a fraction of the gap, not all of it.

## dynamical.org makes this practical

Everything needed is in one anonymous, cloud-optimised catalog (`stac.dynamical.org`),
Icechunk Zarr on S3 `us-west-2`:

| dataset | grid | cadence | coverage | why |
|---|---|---|---|---|
| `noaa-mrms-conus-analysis-hourly` | 0.01° (~1.1 km) CONUS | hourly | **2014-11 → now** | radar truth, `precipitation_surface` |
| `noaa-hrrr-analysis` | 3 km CONUS | hourly | **2014-10 → now** | `composite_reflectivity` + 27 fields |
| `noaa-gfs-analysis` | 0.25° global | hourly | 2021-05 → now | what we ship today |
| `nasa-imerg-analysis` | 0.1° global | 30 min | 1998 → now | global satellite baseline |

Two things this fixes outright:

- **The archive limit is gone.** GFS on S3 retains about two weeks, which is why the
  verification archive has to accumulate slowly. Here there are 102,886 hourly MRMS
  timesteps — 11.7 years — available immediately, so the seasonal refit that was
  blocked on wall-clock time becomes a query.
- **HRRR replaces GFS as the model input.** It is convection-allowing at 3 km with
  composite reflectivity, rather than 28 km GFS whose CSI against observations sits
  near 0.19 regardless of lead. Better input, same pipeline shape.

## The chunking decides the design

MRMS chunks are `(648 time, 100 lat, 100 lon)`. That matters enormously:

| read | time |
|---|---|
| one full field (3500×7000, all lon) | **251 s** |
| 648 hours × one 100×100 tile (one chunk) | **1.0 s** |

A 250x difference. The store is laid out for long time series over small tiles, which
is exactly the shape of a training sample — and exactly the wrong shape for rendering
maps. So: **train from tiles, never from full fields.**

That makes the dataset cheap. ~500 chunk reads is about 8 minutes and yields ~108,000
samples of 100×100 (~111 km) from 11 years of radar; at uint8 with 10 frames per
sample that is ~10 GB, inside Kaggle's 20 GB limit.

## Compute is not the constraint

On Kaggle's free 20 TPU-hours/week, a 20k-sample × 100-epoch run is about 0.66
TPU-hours; 50k × 50 epochs is 0.83. Even generous training is a rounding error against
the weekly budget. Data volume, I/O and careful evaluation are the real costs.

## Gotchas already hit, worth not repeating

- **Optical flow on near-dry tiles returns exactly zero**, which looks like a broken
  motion field but is correct: there is nothing to track. On a genuinely wet tile (56%
  above 0.1 mm/h) the same code gives 22 px/h ≈ 24 km/h. Sample training tiles for rain
  rather than uniformly, or most of the set will be empty sky.
- **Rates are small.** MRMS `precipitation_surface` is kg m⁻² s⁻¹; multiply by 3600 for
  mm/h, and the 99th percentile of a typical tile is ~0.7 mm/h. Any uint8 quantisation
  has to happen after the dBZ transform, not on the raw rate, or nearly everything
  collapses to zero. This is the same class of bug as the float-input flow failure.
- **MRMS here is hourly**, not the 2-minute native feed. Fine for training and for
  1–6 h leads; a true 0–2 h nowcast still wants the raw `noaa-mrms-pds` bucket that
  `verify/radar.py` already reads.

## Step 1 is done, and it changes the picture

Comparing each candidate input against MRMS radar on the same grid at the same hour
(a wet Iowa case, 2026-07-03 09Z, 36% of the box raining):

| input vs radar | 1 mm/h | 4 mm/h | 8 mm/h | POD @1mm/h |
|---|---|---|---|---|
| satellite retrieval (what we ship) | 0.182 | 0.105 | 0.080 | 0.22 |
| **HRRR composite reflectivity** | **0.551** | **0.385** | **0.261** | **0.87** |

**A 3x better input, before any nowcasting at all.** HRRR sees 87% of the rain the
radar sees where the satellite sees 22%. The ceiling that caps the satellite product
is a property of the sensor, and swapping the sensor lifts it.

That reorders everything below. The model is no longer the interesting variable — the
input is. A nowcast built on HRRR starts from roughly where the satellite product
*ends up*, and training then has real headroom to work in rather than fighting a
retrieval that never saw the rain.

The obvious cost is coverage: HRRR is CONUS-only. So this splits the product in two
rather than replacing it —

- **CONUS**: HRRR-based, radar-verified, competitive with published work.
- **Everywhere else**: the satellite blend, which remains the only thing that runs at
  all where there is no radar and no convection-allowing model.

## Order of work

1. ~~**Re-verify with HRRR in place of GFS.**~~ Done, see above: 3x the skill of the
   satellite input against radar truth.
2. **Build the tile dataset**: sample rain-weighted tiles across 11 years, pair MRMS
   truth with HRRR predictors, store uint8, hold out whole months for validation so
   nothing leaks across time.
3. **Train** a sequence model (4 in → 6 out, the DGMR/SEVIR formulation) on Kaggle TPU.
4. **Score it with `verify/run_radar.py`**, against the persistence and advection
   baselines already measured there, at 1/2/4/8 mm/h.

The measured claim to beat is not a published leaderboard — it is our own advection
baseline on radar truth: CSI 0.186 / 0.164 / 0.134 at +30/60/90 min at 1 mm/h. With
HRRR as the input, that bar should be cleared comfortably; the interesting question is
how close the trained model gets to the 0.551 zero-lead ceiling as lead time grows.

One caveat on the numbers above: they are a single wet case in one region. The 3x gap
is far too large to be noise, but the exact figures should be pooled over many cases
before being quoted anywhere. `verify/run_radar.py` already does that pooling for the
satellite path and is the natural place to add an HRRR row.
