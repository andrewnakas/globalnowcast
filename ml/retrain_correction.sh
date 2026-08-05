#!/bin/bash
# Weekly correction retrain, gated. Run on the training box (nakas-1080).
#
# The pair harvest adds ~4 valid times a day, and more data has measurably
# helped once already: 51 pairs -> 77 lifted the mean CSI gain +0.0686 ->
# +0.0809. So the model should be retrained as the archive grows - but only
# promoted when the numbers say so, because "newer" is not "better".
#
# A candidate ships only if it beats the incumbent's recorded gain by more than
# the replicate noise this project measured (two v2 seeds landed +0.0686 and
# +0.0683, so ~0.003). Everything else is logged and discarded.
set -u
cd ~/globalnowcast || exit 1

LOG=~/retrain_status.log
STAMP=$(date -u +%Y-%m-%dT%H:%MZ)
NOISE=0.003
INCUMBENT_FILE=ml/model/.correction_gain
PAIRS=$(ls ml/gfs_pairs/*.npz 2>/dev/null | wc -l | tr -d ' ')

# The incumbent's gain, recorded when it was promoted. Absent on first run.
INCUMBENT=$(cat "$INCUMBENT_FILE" 2>/dev/null || echo 0)

.venv/bin/python -u ml/train_correction.py --out ml/model/refc_correction_cand.pt \
  > /tmp/retrain_cand.log 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  echo "$STAMP rc=$RC pairs=$PAIRS TRAINING FAILED" >> "$LOG"
  exit 1
fi

GAIN=$(grep "best mean CSI gain" /tmp/retrain_cand.log | grep -oE '[+-][0-9.]+' | tail -1)
if [ -z "$GAIN" ]; then
  echo "$STAMP rc=$RC pairs=$PAIRS NO GAIN PARSED" >> "$LOG"
  exit 1
fi

# Promote only on a margin that clears the measured noise floor.
BETTER=$(.venv/bin/python -c "print(1 if $GAIN - $INCUMBENT > $NOISE else 0)")
if [ "$BETTER" = "1" ]; then
  # Export under the name pipeline/correct.py loads, so promoting is a single
  # step. The ONNX is verified against torch inside export_correction.py, and a
  # failure there leaves the incumbent file untouched.
  if .venv/bin/python -u ml/export_correction.py \
       --ckpt ml/model/refc_correction_cand.pt \
       --out ml/model/refc_correction_v3.onnx >> /tmp/retrain_cand.log 2>&1; then
    echo "$GAIN" > "$INCUMBENT_FILE"
    echo "$STAMP pairs=$PAIRS gain=$GAIN vs $INCUMBENT PROMOTED - commit and push" \
      >> "$LOG"
  else
    echo "$STAMP pairs=$PAIRS gain=$GAIN EXPORT FAILED, incumbent kept" >> "$LOG"
  fi
else
  echo "$STAMP pairs=$PAIRS gain=$GAIN vs $INCUMBENT held (margin < $NOISE)" >> "$LOG"
fi
