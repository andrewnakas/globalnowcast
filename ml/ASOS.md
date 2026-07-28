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
