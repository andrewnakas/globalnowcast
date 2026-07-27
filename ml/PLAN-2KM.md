# Competing at 1–2 km: what it takes and what it can honestly claim

The current product is global at 0.25° (~28 km). Published nowcasting benchmarks are
regional at 1–2 km, and [coarse grids inflate CSI](../verify/BENCHMARKS.md) by roughly
+7% per doubling, so our numbers cannot be compared to theirs. This is the plan to
produce numbers that can.

## The source can support it

The satellite blend is natively **0.02° ≈ 2.2 km**, already in the benchmark range —
we currently throw away ~156 pixels per output cell. A radial power spectrum over a
convective window keeps falling from ~570 km down to ~6 km with no flattening, so the
fine structure is real retrieval detail, not interpolation. Resolving to 2 km is
therefore a data-processing change, not a fabrication.

## Global 2 km does not fit; regional does

| grid | dims (70N–60S) | pixels | float32 field |
|------|------|--------|------|
| 0.25° (current) | 521 × 1440 | 0.75 M | 3 MB |
| 0.04° (~4.4 km) | 3251 × 9000 | 29 M | 117 MB |
| 0.02° (~2.2 km) | 6501 × 18000 | **117 M** | **468 MB** |

Optical flow holds ~6 arrays live plus a 2× upscale — about **11 GB peak at global
2 km against the runner's ~7 GB**, and 17 global frames at that size would exceed the
~1 GB Pages artifact limit. Global 2 km is out on free infrastructure.

That is fine, because **no benchmark is global**. DGMR is UK + US radar, NowcastNet is
US/China, SEVIR is 384 × 384 km CONUS tiles. Competing means running *their* domains at
*their* resolution, which is a bounded region — entirely affordable.

## Two separable pieces of work

### A. Regional 2 km nowcast (no ML)

Generalise `pipeline/obs.py` and `pipeline/nowcast.py` from a hardcoded 0.25° target to
a configurable window and resolution. The optical-flow and blend code is
resolution-agnostic already; the coupling is `_reduceat_index`, the module-level
`GFS_LAT/GFS_LON`, and the cached meshgrid in `nowcast.advect`.

Worth noting: **at 2 km the flow pair should be closer together in time, not further.**
The 30-minute pair was chosen because 10 minutes is sub-pixel at 28 km. At 2 km a
10-minute gap is ~14 px of motion, so the shortest-latency pair becomes usable and the
nowcast gets fresher. This needs re-measuring, not assuming.

### B. A trained model (Kaggle TPU)

Kaggle's free 20 TPU-hours/week is more than enough — **compute is not the binding
constraint**:

| samples | epochs | TPU-hours (v3-8, 30% util) |
|---------|--------|------|
| 20 000 | 50 | 0.33 |
| 20 000 | 100 | 0.66 |
| 50 000 | 50 | 0.83 |

The constraints are data volume and I/O:

| tile | samples | float16 | uint8 |
|------|---------|---------|-------|
| 256² (~568 km) | 20 000 | 26 GB | **13 GB** |
| 256² | 50 000 | 66 GB | 33 GB |
| 512² | 20 000 | 105 GB | 52 GB |

**256² tiles × 20 000 samples, stored uint8 (dBZ quantised to 0.5), is 13 GB** — inside
Kaggle's 20 GB dataset limit, and 256² is the tile size SEVIR and most nowcasting papers
train on. Building it needs ~500 global frames (~3.5 days of observations, ~2 GB
download, ~15 min at 8-way parallelism), tiled preferentially over convective areas.

This is a different model from `ml/model.py`. That one is a 59 k-param residual UNet
that de-biases a single GFS frame. This is sequence-to-sequence: 4 input frames → 6
output frames, which is the DGMR/NowcastNet/SEVIR formulation.

## Scoring: use radar, not ourselves

Verifying satellite forecasts against later satellite observations measures internal
consistency and shares the retrieval's own errors. The benchmarks all verify against
**radar**. `ml/data/mrms.py` already fetches MRMS (CONUS, ~1 km, 2-min, anonymous S3),
so scoring the 2 km CONUS product against MRMS is a small step and is the comparison
that actually rebuts the resolution objection.

## What this can and cannot claim

It **can** produce CSI at 1/4/8 mm/h on a ~2 km CONUS grid verified against radar —
directly comparable in construction to DGMR and NowcastNet.

It **cannot** be assumed to beat them, and the plan should not pretend otherwise. Those
models are trained on radar and predict radar; ours observes rain rate from infrared
and microwave, which is an inherently blurrier, lagged view of precipitation —
especially for convective initiation, which satellites see late. The honest expectation
is that we are **competitive at light thresholds and longer leads, and behind at high
thresholds and short leads**, where radar's direct view of hydrometeors wins.

The genuine claim is different and still strong: this works **anywhere on Earth**,
including the ~70% of land with no radar coverage at all, where DGMR and NowcastNet
cannot run. That is worth measuring precisely rather than overselling.

## Order of work

1. Make the target grid configurable; verify the 0.25° global product is bit-identical.
2. Stand up a 2 km CONUS window; re-measure the optimal flow-pair separation.
3. Score it against MRMS radar at 1/4/8 mm/h — the first genuinely comparable number.
4. Only then build the tile dataset and train, with step 3 as the baseline to beat.

Step 3 is the deliverable that answers "how do we compare". Steps 1–3 need no GPU and
no ML at all, so they should land before any training work starts.
