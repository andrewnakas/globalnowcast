# Overnight run, 2026-08-07

Production is unchanged tonight, and that is the headline: four experiments
were run to decide whether to change it, and all four said no. The work that
did ship was repair — three variants of the same silent-failure bug, each of
which was throwing away data while reporting success.

## Nothing shipped to the forecast

| question | verdict | numbers |
|---|---|---|
| AIFS inside the blend window | **no** | mean −0.0007 over 20 lead×threshold cells, 3/20 improved |
| GEFS as a third model arm | **no** | mean +0.0031 but −0.0089 at 30 dBZ (3× noise) |
| Regional handover split | **not yet** | tropics 420 / midlat 330, margins +0.0011 / +0.0005 (inside noise) |
| CONUS 120-min crossover | **holds** | advection +0.035 ahead at +60m, −0.022 behind at +120m |

The AIFS in-window case is the instructive one. At 5 dBZ alone it looks like a
+10.9% win at +270 min, and the model arm rises from 0.148 to 0.19. Across all
four thresholds it is a net loss, because it is the same trade AIFS always
makes — better at light rain, worse at heavy — and it loses at 20 dBZ at every
lead. This project has been caught by a single-threshold crossover conclusion
before; the all-cells ranking exists because of it.

GEFS was genuinely borderline: 3 of 4 thresholds improved and a mean-only rule
would have shipped it. It was declined because the loss concentrates at 30 dBZ,
the shipped two-model mean *already* spends 6% of heavy-rain skill to buy its
light-rain gains, and compounding that degrades exactly what a radar map is
consulted for. GEFS also initialises every 24 h against GFS's 6 h, so these
numbers are its best case rather than its typical one.

## What was broken, and is now fixed

**The verification archive had been frozen at 20 cases for three weeks.** The
weekly job ran, scored cases, reported success, and discarded everything. When
`archive_conus.jsonl` was added to the commit step it named a path git had
never seen, so `git diff --quiet -- <nonexistent>` failed and killed the step.
Adding CONUS accumulation broke global accumulation. Now 35 global cases and
the first 8 CONUS cases this project has ever kept.

**The CONUS harness was advecting the wrong field.** Production advects MRMS;
`verify/run_radar.py` advected satellite, which carries a CSI-0.18 sensor
ceiling against radar. It put advection at 0.177 against HRRR's 0.215 at +1 h —
the reverse of the 0.282 vs 0.176 the crossover was fitted on. Refitting the
handover from that archive, which is exactly what tonight's priorities asked
for, would have moved it hours early and degraded the layer while appearing to
improve it. The eight bad cases are quarantined in `archive_conus_satadv.jsonl`
rather than deleted.

**A third variant of the same loss.** `verify/run.py` and `archive.py` exit
non-zero when they score nothing, which is right interactively and wrong in the
weekly job: it fails the run and skips the commit step, discarding cases the
other steps *had* scored. Both scoring steps and the summary are now
`continue-on-error`; the commit step stays strict, because silence there is
what caused the original three-week loss.

## Automation now proven rather than assumed

Both branches of the Monday gated retrain were exercised end-to-end:

- **hold** — candidate inside the ±0.003 noise floor, incumbent kept
- **promote** — candidate beats incumbent, exports a valid 255 KB ONNX, updates
  the incumbent record

An earlier version of that test mutated `ml/model/.correction_gain` directly,
and an ssh drop left it at a sabotage value that would have silently blocked
every future promotion. The tests now write only to `/tmp`. Test harnesses must
never point at production state.

## Two open hints, deliberately not acted on

1. **Regional handover split** (tropics 420 / midlat 330) is consistent across
   20 and 33 cases and physically sensible — less-organised tropical convection
   means extrapolation holds value longer. Margins are inside the noise floor.
   Decidable at ~60 cases, roughly a fortnight at 13/week.
2. **Intensity-dependent CONUS handover**: at 4 mm/h HRRR is already ahead at
   +60 m (0.139 vs 0.115), where at 1 mm/h advection still leads. Eight cases
   cannot support that change.

## Corrections to my own claims

- Commit 256173e quoted a lead-by-lead CONUS table as though pooled over cases.
  It was **one case**; the rest of that run died on local DNS failures. Noted in
  `verify/run_radar.py` and in d9bf8b6.
- I told the user the AIFS blend made a crossover refit mandatory. It did not:
  the handover is fitted at leads 30–270 min, entirely inside the window where
  production keeps raw GFS, and AIFS only enters past +6 h.
- The ASOS gauge re-run scored far below the 2026-07-29 run, which looked like a
  regression and was not: no CONUS-path file had changed, and the radar *truth*
  dropped too (0.302 → 0.252) with gauge bias rising 2.21 → 2.49×. Compare runs
  by retention (blend CSI / radar CSI), not raw CSI.

## State at hand-off

Site fresh and serving all layers (`corrected: true`, `multimodel: true`, 63
frames). Hourly builds green. 85 correction pairs accumulating at +4/day, both
crons armed, incumbent `+0.0809`.
