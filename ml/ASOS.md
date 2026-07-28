# ASOS station observations: verification, not training

`dynamical.org/catalog/asos-parquet` carries hourly surface observations from the Iowa
Environmental Mesonet, 1940 to present, as year-partitioned GeoParquet at
`https://data.source.coop/dynamical/asos-parquet/year={YYYY}/data.parquet`. Anonymous,
queryable directly with DuckDB's httpfs.

Checked against 2026 so far, CONUS only:

| | |
|---|---|
| stations | 2,454 |
| observations | 24.1 M |
| with `p01m` (1 h precip, mm) | all of them |
| hours with >0.2 mm | 831 k |

## Why this is worth having

Everything in this project so far is verified against MRMS, and MRMS is itself a radar
product. Radar does not measure rain - it measures backscatter from hydrometeors aloft
and converts that to a rate through an assumed Z-R relationship, the same
`Z = 200 R^1.6` this project applies to satellite. Verifying a radar-trained model
against radar shares every one of those assumptions.

ASOS rain gauges measure water actually landing in a bucket at the surface. They are
sparse, they undercatch in wind, and heated tipping buckets lag in cold events - but
their errors are unrelated to radar's. A gain that shows up against both is real; a
gain that shows up only against MRMS may be the model learning the radar's quirks.

That makes ASOS the answer to a question BENCHMARKS.md raises and cannot currently
settle: whether +22.7% over HRRR reflects better forecasting or better agreement with
one particular instrument.

## Why not train on it

Two reasons, both structural rather than a matter of effort.

It is point data. The model predicts a 100x100 field and would have supervision at a
few dozen scattered pixels per tile - useful as a check, far too sparse as a target.
Gridding it first would mean interpolating between stations hundreds of kilometres
apart, which invents exactly the spatial structure the model is supposed to predict.

And it is hourly accumulation, not instantaneous rate. The sub-hourly work is aimed at
10-minute steps, which ASOS cannot supervise at all.

## First result: radar and gauges disagree a lot

Running `verify/run_asos.py` over three hours, ~2,490 stations each:

| threshold | CSI | POD | FAR | bias |
|-----------|-----|-----|-----|------|
| 0.2 mm | 0.2655 | 0.681 | 0.697 | 2.25 |
| 1 mm | 0.2075 | 0.543 | 0.749 | 2.16 |
| 4 mm | 0.0870 | 0.250 | 0.882 | 2.12 |
| 8 mm | 0.1379 | 0.286 | 0.789 | 1.36 |

That is MRMS against gauges, with no model involved. Radar reports rain over about
2.2x the area gauges measure it, and at 4 mm nearly nine in ten radar exceedances have
no gauge behind them.

Some of that gap is the comparison rather than the instruments - a gauge is a funnel a
few inches across and a radar cell is 2 km, so a cell can legitimately average above a
threshold that the one point inside it never reaches. But the size of it is a caution:
**a radar-trained model cannot score better against gauges than its own training
target does.** CSI 0.27 is the ceiling for anything learned from MRMS, whatever it
scores against MRMS itself.

This does not invalidate the +22.7% over HRRR. That comparison is model against model
on identical truth, and it stands. What it does mean is that "CSI 0.51" is a statement
about agreement with radar, not about how much rain the forecast gets right at the
ground, and those are further apart than they look.

## How to use it

Score the existing model's +1 h forecast at station locations, against `p01m`, and
compare the ranking to the MRMS ranking:

- model, HRRR and persistence all improve or worsen together -> MRMS verification is
  trustworthy and the numbers stand;
- the model's lead shrinks or reverses -> it has partly learned radar-specific
  behaviour, and the headline needs qualifying.

The comparison is only meaningful with matched units and timing. `p01m` is an hour's
accumulation ending at the observation time, while the model predicts an instantaneous
rate at that time, so the model field has to be accumulated over the hour before the
two are comparable. Skipping that step would make the model look wrong for a reason
that has nothing to do with its skill.
