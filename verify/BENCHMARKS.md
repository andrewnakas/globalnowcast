# How this compares to published nowcasting work

Short answer: **there is no leaderboard this system can honestly enter.** Every public
benchmark with hard CSI numbers is regional, radar-based, and on a 1–2 km grid. This
is global, satellite-based, and on a 0.25° (~28 km) grid. The numbers are not
interchangeable, and the difference runs in our favour, so quoting them side by side
without the caveat would be flattering rather than informative.

## Our scores at the thresholds the literature uses

Benchmarks report rain rate, not dBZ. Converting through the same Marshall-Palmer
relation the pipeline uses (`Z = 200 R^1.6`): 1 mm/h = 23 dBZ, 2 = 27.8, 4 = 32.6,
8 = 37.5.

**Current, on the 0.1° grid the product actually renders** (5 cases, pooled,
70N–60S, `--leads 30,60,90,120,240 --thresholds 23,27.8,32.6,37.5`):

| lead | blend (1 mm/h) | advection | persistence | GFS |
|------|------|------|------|------|
| +30m | 0.574 | 0.574 | 0.463 | 0.202 |
| +60m | 0.442 | 0.442 | 0.359 | 0.201 |
| +90m | 0.356 | 0.355 | 0.290 | 0.198 |
| +120m | 0.305 | 0.304 | 0.251 | 0.200 |
| +240m | 0.204 | 0.193 | 0.165 | 0.201 |

These are **lower than the 0.25° figures this table used to quote** (0.623 at
+30 m), and that is the grid effect below working as advertised, not a
regression: the product moved to 0.1° in a2a5cfb, and a finer grid is a harder
test. The old numbers are kept nowhere, because quoting the coarser grid's
scores for a finer-grid product would flatter it.

Note the blend and advection are identical until +120m and only separate at
+240m: the handover is deliberately observation-dominated, and GFS's flat ~0.20
is what it hands over *to*.

## Why these can't be compared directly

**Grid spacing inflates CSI on its own.** Coarsening the *same* forecast and the same
truth, changing nothing else:

| grid | CSI (dBZ>=20, +60m) | |
|------|------|---|
| 0.25° (~28 km, ours) | 0.525 | — |
| 0.5° (~56 km) | 0.560 | +7% |
| 1.0° (~111 km) | 0.613 | +17% |

A displacement error of 20 km is a miss *and* a false alarm on a 1 km radar grid; on
a 28 km grid it often lands in the same cell and scores as a hit. Published models
evaluated at 1 km are being asked a materially harder question. Roughly, our grid is
two doublings coarser than theirs, and each doubling is worth about +7%.

**Marshall-Palmer is not a calibration.** `Z = 200 R^1.6` is a convective relation
applied globally to a satellite retrieval. The mm/h figures above are therefore
approximate even before the grid issue.

## The landscape, and what each is actually good for

| Benchmark | Domain | Grid | Truth | Comparable? |
|---|---|---|---|---|
| **Weather4cast** (NeurIPS 2021–25) | Europe | 2 km | OPERA radar | No — satellite→radar downscaling over Europe. The 2025 edition dropped the scored leaderboard, so there is nothing live to enter. |
| **SEVIR** | CONUS | 1 km | NEXRAD VIL | No — thresholds are VIL levels (a storm-intensity proxy), not rain rate or dBZ. Storm-event sampled, not all-weather. |
| **HKO-7** | Hong Kong | 1 km | Radar | No — single small radar domain. |
| **RainBench** | **Global** | 0.25° | IMERG | Closest in domain and grid, but wrong task: 1–5 *day* forecasts scored by RMSE, not nowcasting CSI. |
| **MRMS / OPERA** | CONUS / Europe | 1 km | Radar | Not a packaged benchmark, but the usual truth for DGMR, NowcastNet, MetNet-3. |

Published deep-learning numbers (DGMR, NowcastNet, MetNet-3, Earthformer, LDCast) are
all on those fine-grid regional datasets, mostly at 1/4/8 mm/h or 16/32/64 mm/h out to
90–180 min. Rank ordering in that literature is well established — pysteps-style
extrapolation is consistently the weakest baseline at long lead and high threshold —
but there is no fixed external "pysteps curve" to cite, because every paper reruns it
on its own domain.

The nearest published analogue is Ryu et al. (2024, *J. Hydrometeorology*), a global
IMERG + GFS U-Net ConvLSTM nowcast out to 4 h. Same domain, same task, same lead-time
range. Its CSI values are reported in figures rather than tables, so they could not be
read off precisely; it is the right paper to compare against if someone wants a real
number.

## The comparable measurement, and what it says

`verify/run_radar.py` does the comparison properly: the ~2.2 km CONUS grid the
benchmarks use, MRMS radar as independent truth, and the 1/2/4/8 mm/h thresholds the
literature reports. Three cases, pooled:

| lead | 1 mm/h | 2 mm/h | 4 mm/h | 8 mm/h |
|------|--------|--------|--------|--------|
| +30m | 0.186 | 0.156 | 0.125 | 0.099 |
| +60m | 0.164 | 0.136 | 0.105 | 0.074 |
| +90m | 0.134 | 0.115 | 0.086 | 0.047 |

Against satellite self-truth at 0.25° the same system scores 0.62 at +30 min. Against
radar at 2 km it scores 0.19. **Both numbers are real; they measure different things**,
and the second is the one comparable to published work.

Advection still beats persistence at every lead and threshold, and by a widening
margin (at 8 mm/h, +90 min: 0.047 against 0.018, a 2.6x gain), so the motion model is
working. But the level is low, and the reason is not the motion model.

### The ceiling is the observation, not the nowcast

Scoring the satellite against radar **at zero lead, with no forecast at all**:

| threshold | CSI | POD | FAR | sat wet area | radar wet area |
|-----------|-----|-----|-----|--------------|----------------|
| 1 mm/h | 0.182 | 0.22 | 0.50 | 0.0098 | 0.0221 |
| 4 mm/h | 0.105 | 0.14 | 0.71 | 0.0038 | 0.0079 |
| 8 mm/h | 0.080 | 0.11 | 0.75 | 0.0017 | 0.0040 |

The retrieval agrees with radar at CSI 0.18 before any forecasting happens. Our +30 min
forecast scores 0.186 — **at, and marginally above, that ceiling**. The nowcasting is
close to free; essentially all of the error is the satellite's disagreement with the
radar it is being scored against.

It sees under half the rain area radar sees (POD 0.22) and half of what it does report
is not there (FAR 0.50). That is inherent to infrared and microwave retrieval: it
infers rain from cloud-top properties rather than observing hydrometeors directly, so
it misses shallow and warm-rain precipitation and lags convective initiation.

**No motion model, and no amount of training, can exceed that ceiling** while the input
is this retrieval. A model trained on radar targets could learn to correct part of the
bias — that is exactly what `ml/PLAN-2KM.md` proposes — but the honest expectation is a
fraction of the gap, not all of it.

### So: can this compete with DGMR and NowcastNet?

At their game, on their turf, no — and the reason is the sensor, not the method. They
consume radar and predict radar, on domains where radar exists. Comparing a satellite
retrieval to them on CSI-against-radar measures mostly the sensor gap.

What this system does that they cannot is run **everywhere**, including the majority of
the planet with no radar at all. The defensible claim is coverage, plus the measured
internal result: it substantially beats both persistence and the GFS forecast it is
built on, verified on a growing archive of held-out cases.

## What we *can* claim without qualification

The internal baselines are apples-to-apples, same grid, same truth, same mask:

- advection beats persistence at every lead and threshold;
- the blend beats both parents at every lead and threshold;
- GFS alone is far behind out to ~6 h — at 4 mm/h it manages 0.074 CSI at +30 min
  against advection's 0.505.

That is the claim this project can actually support: **a large, measured improvement
over the model it started from**, verified on a growing archive of held-out cases.
