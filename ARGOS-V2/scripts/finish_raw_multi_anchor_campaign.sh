#!/usr/bin/env bash
set -euo pipefail

root=/dtu/p1/leopam/ARGOS/ARGOS-V2
output="$root/results/raw_multi_anchor_temporal_refiner"
python=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
runner="$root/scripts/run_raw_multi_anchor_temporal_refiner.py"
mkdir -p "$output/logs" /tmp/leopam_argos_v2
export TMPDIR=/tmp/leopam_argos_v2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$root"

wait_for_pidfile() {
  local pidfile=$1 checkpoint=$2 label=$3
  until test -s "$pidfile"; do sleep 30; done
  local pid
  pid=$(cat "$pidfile")
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  if ! test -s "$checkpoint"; then
    echo "$label ended without $checkpoint" >&2
    exit 1
  fi
  echo "$label complete: $checkpoint"
}

run_gpu() {
  local gpu=$1 log=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$python" "$runner" "$@" \
    > "$output/logs/$log" 2>&1
}

# Training was launched before this orchestrator. The CS1 watcher starts its
# job only after soft training releases GPU 1.
wait_for_pidfile "$output/logs/soft_train.pid" "$output/soft_fusion/checkpoints/best_validation.pt" soft
wait_for_pidfile "$output/logs/hard_train.pid" "$output/hard_retrieval/checkpoints/best_validation.pt" hard
wait_for_pidfile "$output/logs/cs1_train.pid" "$output/cs1_reference/checkpoints/best_validation.pt" cs1

# Freeze learned policies and the deterministic consensus policy on dataset 2.
run_gpu 0 hard_calibrate.log --mode calibrate --configuration hard --output "$output" --device cuda:0 --flow-batch-size 1 --batch-size 8 --workers 16 & p0=$!
run_gpu 1 soft_calibrate.log --mode calibrate --configuration soft --output "$output" --device cuda:0 --flow-batch-size 4 --batch-size 8 --workers 16 & p1=$!
wait "$p0"; wait "$p1"
run_gpu 0 consensus_calibrate.log --mode calibrate --configuration consensus --output "$output" --device cuda:0 --flow-batch-size 1 --batch-size 8 --workers 16 & p0=$!
run_gpu 1 cs1_calibrate.log --mode calibrate --configuration cs1 --output "$output" --device cuda:0 --flow-batch-size 4 --batch-size 8 --workers 16 & p1=$!
wait "$p0"; wait "$p1"

# Validation reporting uses dataset 2 only. No dataset-7 path is touched here.
run_gpu 0 hard_validation.log --mode evaluate --configuration hard --split validation --output "$output" --device cuda:0 --flow-batch-size 1 --batch-size 8 --workers 16 & p0=$!
run_gpu 1 soft_validation.log --mode evaluate --configuration soft --split validation --output "$output" --device cuda:0 --flow-batch-size 4 --batch-size 8 --workers 16 & p1=$!
wait "$p0"; wait "$p1"
run_gpu 0 consensus_validation.log --mode evaluate --configuration consensus --split validation --output "$output" --device cuda:0 --flow-batch-size 1 --batch-size 8 --workers 16 & p0=$!
run_gpu 1 cs1_validation.log --mode evaluate --configuration cs1 --split validation --output "$output" --device cuda:0 --flow-batch-size 4 --batch-size 8 --workers 16 & p1=$!
wait "$p0"; wait "$p1"

# TTL-2 is allowed only when the primary raw bank improves the fixed H=4
# reference on validation. This decision is made before dataset 7 is opened.
if "$python" - "$output/soft_fusion/validation/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
raise SystemExit(0 if summary["gain_over_fixed_h4"] > 0 else 1)
PY
then
  run_gpu 1 hybrid_train.log --mode train --configuration hybrid_ttl2 --output "$output" --device cuda:0 --flow-batch-size 4 --batch-size 8 --workers 16 --channels 32 --blocks 3 --epochs 10
  run_gpu 1 hybrid_calibrate.log --mode calibrate --configuration hybrid_ttl2 --output "$output" --device cuda:0 --flow-batch-size 4 --batch-size 8 --workers 16
  run_gpu 1 hybrid_validation.log --mode evaluate --configuration hybrid_ttl2 --split validation --output "$output" --device cuda:0 --flow-batch-size 4 --batch-size 8 --workers 16
fi

# All model/checkpoint/policy choices are now frozen. Each configuration opens
# dataset 7 exactly once for its final report.
run_gpu 0 hard_test.log --mode evaluate --configuration hard --split test --output "$output" --device cuda:0 --flow-batch-size 1 --batch-size 8 --workers 16 & p0=$!
run_gpu 1 soft_test.log --mode evaluate --configuration soft --split test --output "$output" --device cuda:0 --flow-batch-size 4 --batch-size 8 --workers 16 & p1=$!
wait "$p0"; wait "$p1"
run_gpu 0 consensus_test.log --mode evaluate --configuration consensus --split test --output "$output" --device cuda:0 --flow-batch-size 1 --batch-size 8 --workers 16 & p0=$!
run_gpu 1 cs1_test.log --mode evaluate --configuration cs1 --split test --output "$output" --device cuda:0 --flow-batch-size 4 --batch-size 8 --workers 16 & p1=$!
wait "$p0"; wait "$p1"
if test -s "$output/hybrid_ttl2/frozen_policy.json"; then
  run_gpu 1 hybrid_test.log --mode evaluate --configuration hybrid_ttl2 --split test --output "$output" --device cuda:0 --flow-batch-size 4 --batch-size 8 --workers 16
fi

"$python" "$runner" --mode summarize --configuration soft --output "$output" \
  > "$output/logs/summarize.log" 2>&1
echo "ARGOS v2 raw multi-anchor campaign complete"
