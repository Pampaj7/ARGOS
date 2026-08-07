#!/usr/bin/env bash
set -euo pipefail
root=/dtu/p1/leopam/ARGOS/ARGOS-V2
soft_pid="$(cat "$root/results/raw_multi_anchor_temporal_refiner/logs/soft_train.pid")"
while kill -0 "$soft_pid" 2>/dev/null; do sleep 30; done
if ! test -f "$root/results/raw_multi_anchor_temporal_refiner/soft_fusion/checkpoints/best_validation.pt"; then
  echo "soft training ended without a best checkpoint; refusing dependent launch" >&2
  exit 1
fi
nohup "$root/scripts/launch_raw_multi_anchor_cs1_gpu1.sh" \
  > "$root/results/raw_multi_anchor_temporal_refiner/logs/cs1_train.log" 2>&1 &
echo $! > "$root/results/raw_multi_anchor_temporal_refiner/logs/cs1_train.pid"
