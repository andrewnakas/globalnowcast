#!/usr/bin/env bash
# Run an evaluation only when the box can afford it.
#
# Evaluations load a few thousand sequences on top of a trainer already holding ~7 GB
# of 16 GB. Running them concurrently pushed the trainer into swap and left it in
# uninterruptible I/O sleep for half an hour - it recovered, but it stopped training
# while it thrashed. Better to wait for the memory than to race it.
#
#   ./ml/eval_when_free.sh ml/model/nowcast_all32_s1.pt ml/tiles_all
set -u
cd "$(dirname "$0")/.."
CKPT="${1:?checkpoint}"; TILES="${2:-ml/tiles_all}"; NEED_GB="${3:-4}"
while true; do
  free_gb=$(free -g | awk "/^Mem:/ {print \$7}")
  [ "$free_gb" -ge "$NEED_GB" ] && break
  echo "waiting for memory: ${free_gb}G available, need ${NEED_GB}G"
  sleep 120
done
.venv/bin/python ml/eval_nowcast.py --ckpt "$CKPT" --tiles "$TILES" --val-months 12
.venv/bin/python ml/eval_pm.py     --ckpt "$CKPT" --tiles "$TILES" --val-months 12
