#!/usr/bin/env bash
# Run the nowcast experiment queue back to back, unattended.
#
# Each entry is a full train + evaluate. Results land in ml/model/ and one line per
# run goes to ml/sweep_results.tsv, so an interrupted sweep can be read at any point
# rather than only at the end.
#
# The queue is ordered so the cheap, most-likely-to-matter experiments run first:
# more data beat every hyperparameter change so far, and gated fusion is the
# architecture change the ablation literature rates highest.
#
#   ./ml/sweep.sh ml/tiles_big
set -u
cd "$(dirname "$0")/.." || exit 1
TILES="${1:-ml/tiles_big}"
PY=.venv/bin/python
OUT=ml/sweep_results.tsv
mkdir -p ml/model
[ -f "$OUT" ] || printf 'name\tbase\tgated\twet\tepochs\tbest_csi\thrrr_csi\tgain_pct\n' > "$OUT"

run () {  # name base gated wet epochs
  local name=$1 base=$2 gated=$3 wet=$4 ep=$5
  local ckpt="ml/model/${name}.pt" log="/tmp/sweep_${name}.log"
  if grep -q "^${name}	" "$OUT" 2>/dev/null; then
    echo "== ${name}: already done, skipping"; return
  fi
  local flag=""; [ "$gated" = "1" ] && flag="--gated"
  echo "== ${name}: base=${base} gated=${gated} wet=${wet} epochs=${ep}"
  $PY -u ml/train_nowcast.py --tiles "$TILES" --base "$base" $flag \
      --wet-weight "$wet" --epochs "$ep" --batch 32 --val-months 10 \
      --out "$ckpt" > "$log" 2>&1
  local line best hrrr gain
  line=$(grep '^best val' "$log" | tail -1)
  best=$(echo "$line" | sed -n 's/.*CSI \([0-9.]*\).*/\1/p')
  hrrr=$(echo "$line" | sed -n 's/.*HRRR \([0-9.]*\).*/\1/p')
  gain=$(echo "$line" | sed -n 's/.*(\([-+][0-9.]*\)%.*/\1/p')
  if [ -z "$best" ]; then
    echo "   FAILED - see $log"; tail -3 "$log"
    printf '%s\t%s\t%s\t%s\t%s\tFAILED\t\t\n' "$name" "$base" "$gated" "$wet" "$ep" >> "$OUT"
    return
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$base" "$gated" "$wet" "$ep" "$best" "$hrrr" "$gain" >> "$OUT"
  echo "   best CSI ${best} vs HRRR ${hrrr} (${gain}%)"
}

# name              base gated wet epochs
run gated64          64   1     8   60   # the architecture A/B against the baseline
run big96            96   0     8   60   # does more capacity help now the data is 5x?
run gated96          96   1     8   60   # both, if either wins alone
run gated64_w12      64   1    12   60   # push harder on rain now PM fixes the bias
run gated64_long     64   1     8  120   # was it still improving at 60 epochs?

echo
echo "=== sweep complete ==="
column -t "$OUT" 2>/dev/null || cat "$OUT"
