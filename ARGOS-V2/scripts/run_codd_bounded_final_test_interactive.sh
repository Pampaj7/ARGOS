#!/usr/bin/env bash
# ARGOS v2 one-time frozen dataset-7 evaluation after dataset-2 policy freeze.
set -euo pipefail

ARGOS_ROOT=/dtu/p1/leopam/ARGOS/ARGOS-V2
ARGOS_PYTHON=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
RESULT_ROOT="$ARGOS_ROOT/results/codd_style_bounded_memory_validation"
RUNNER="$ARGOS_ROOT/scripts/run_codd_style_bounded_memory_validation.py"
ABLATION_ROOT="$RESULT_ROOT/ablations"
cd "$ARGOS_ROOT"

while [[ ! -f "$RESULT_ROOT/validation_policy_freeze.timestamp" ]]; do
  sleep 20
done

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" "$RUNNER" \
  --mode evaluate --split test --device cuda:0 --workers 20 --preload-workers 20 \
  --frozen-policy "$RESULT_ROOT/reset_policy/final_reset_policy.json" \
  --output "$RESULT_ROOT/reset_policy/final_frozen_test" \
  >"$RESULT_ROOT/reset_policy/final_frozen_test.log" 2>&1 & pid0=$!
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" "$RUNNER" \
  --mode evaluate --split test --device cuda:0 --workers 20 --preload-workers 20 \
  --frozen-policy "$ABLATION_ROOT/hard_endpoint/final_hard_policy.json" \
  --output "$ABLATION_ROOT/hard_endpoint/test" \
  >"$ABLATION_ROOT/hard_endpoint/test.log" 2>&1 & pid1=$!
wait "$pid0"
wait "$pid1"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" "$RUNNER" \
  --mode evaluate --split test --device cuda:0 --workers 20 --preload-workers 20 \
  --checkpoint "$ABLATION_ROOT/no_recurrence/checkpoints/best_validation.pt" \
  --frozen-policy "$RESULT_ROOT/reset_policy/fixed_h4_policy.json" --memory-state raw_previous \
  --output "$ABLATION_ROOT/no_recurrence/test" \
  >"$ABLATION_ROOT/no_recurrence/test.log" 2>&1 & pid0=$!
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" "$RUNNER" \
  --mode evaluate --split test --device cuda:0 --workers 20 --preload-workers 20 \
  --checkpoint "$ABLATION_ROOT/no_learned_stereo_evidence/checkpoints/best_validation.pt" \
  --frozen-policy "$RESULT_ROOT/reset_policy/fixed_h4_policy.json" --disable-learned-stereo-evidence \
  --output "$ABLATION_ROOT/no_learned_stereo_evidence/test" \
  >"$ABLATION_ROOT/no_learned_stereo_evidence/test.log" 2>&1 & pid1=$!
wait "$pid0"
wait "$pid1"
date -Is >"$RESULT_ROOT/final_test_complete.timestamp"
