#!/usr/bin/env bash
# ARGOS v2: execute the two trainable bounded-memory ablations inside the
# already active two-H100 interactive allocation.
set -euo pipefail

ARGOS_ROOT=/dtu/p1/leopam/ARGOS/ARGOS-V2
ARGOS_PYTHON=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
RESULT_ROOT="$ARGOS_ROOT/results/codd_style_bounded_memory_validation"
SMOKE_ROOT="$RESULT_ROOT/_temporary_smoke"
ABLATION_ROOT="$RESULT_ROOT/ablations"

cd "$ARGOS_ROOT"
mkdir -p "$SMOKE_ROOT" "$ABLATION_ROOT/no_recurrence" "$ABLATION_ROOT/no_learned_stereo_evidence"

run_pair() {
  local mode=$1
  local root=$2
  local extra=$3
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" scripts/run_codd_style_fusion_probe.py \
    --mode "$mode" --seed 0 --device cuda:0 --workers 20 --preload-workers 20 \
    --output "$root/no_recurrence" $extra \
    >"$root/no_recurrence_${mode}.log" 2>&1 &
  local pid0=$!
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" scripts/run_codd_style_fusion_probe.py \
    --mode "$mode" --seed 0 --device cuda:0 --workers 20 --preload-workers 20 \
    --output "$root/no_learned_stereo_evidence" --disable-learned-stereo-evidence \
    >"$root/no_learned_stereo_evidence_${mode}.log" 2>&1 &
  local pid1=$!
  wait "$pid0"
  wait "$pid1"
}

run_pair smoke "$SMOKE_ROOT" "--memory-state raw_previous"
test -f "$SMOKE_ROOT/no_recurrence/smoke_summary.json"
test -f "$SMOKE_ROOT/no_learned_stereo_evidence/smoke_summary.json"
rm -rf "$SMOKE_ROOT/no_recurrence" "$SMOKE_ROOT/no_learned_stereo_evidence"

run_pair overfit "$SMOKE_ROOT" "--memory-state raw_previous"
test -f "$SMOKE_ROOT/no_recurrence/overfit_summary.json"
test -f "$SMOKE_ROOT/no_learned_stereo_evidence/overfit_summary.json"
rm -rf "$SMOKE_ROOT/no_recurrence" "$SMOKE_ROOT/no_learned_stereo_evidence"

run_pair train "$ABLATION_ROOT" "--memory-state raw_previous"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" scripts/run_codd_style_fusion_probe.py \
  --mode evaluate --seed 0 --device cuda:0 --workers 20 --preload-workers 20 \
  --output "$ABLATION_ROOT/no_recurrence" --memory-state raw_previous \
  >"$ABLATION_ROOT/no_recurrence_evaluate.log" 2>&1 & pid0=$!
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" scripts/run_codd_style_fusion_probe.py \
  --mode evaluate --seed 0 --device cuda:0 --workers 20 --preload-workers 20 \
  --output "$ABLATION_ROOT/no_learned_stereo_evidence" --disable-learned-stereo-evidence \
  >"$ABLATION_ROOT/no_learned_stereo_evidence_evaluate.log" 2>&1 & pid1=$!
wait "$pid0"
wait "$pid1"

date -Is >"$RESULT_ROOT/ablation_training_complete.timestamp"
