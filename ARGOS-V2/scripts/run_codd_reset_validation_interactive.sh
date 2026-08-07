#!/usr/bin/env bash
# ARGOS v2 dataset-2-only reset/hard policy calibration on two H100s.
set -euo pipefail

ARGOS_ROOT=/dtu/p1/leopam/ARGOS/ARGOS-V2
ARGOS_PYTHON=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
RESULT_ROOT="$ARGOS_ROOT/results/codd_style_bounded_memory_validation"
VALIDATION_ROOT="$RESULT_ROOT/reset_policy/validation_candidates"
HARD_ROOT="$RESULT_ROOT/ablations/hard_endpoint/validation_candidates"
RUNNER="$ARGOS_ROOT/scripts/run_codd_style_bounded_memory_validation.py"
cd "$ARGOS_ROOT"

while [[ ! -f "$RESULT_ROOT/ablation_training_complete.timestamp" ]]; do
  sleep 20
done

SMOKE_DIR="$RESULT_ROOT/_temporary_policy_smoke"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" "$RUNNER" \
  --mode evaluate --split validation --tiny --device cuda:0 --workers 2 --preload-workers 2 \
  --policy-name smoke_h4 --max-age 4 --output "$SMOKE_DIR" \
  >"$RESULT_ROOT/policy_smoke.log" 2>&1
test -f "$SMOKE_DIR/summary.json"
rm -rf "$SMOKE_DIR"

run_candidate() {
  local gpu=$1
  local name=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" "$RUNNER" \
    --mode evaluate --split validation --device cuda:0 --workers 20 --preload-workers 20 \
    --policy-name "$name" --output "$VALIDATION_ROOT/$name" "$@" \
    >"$VALIDATION_ROOT/${name}.log" 2>&1
}

mkdir -p "$VALIDATION_ROOT" "$HARD_ROOT"
(
  run_candidate 0 fixed_h1 --max-age 1
  run_candidate 0 fixed_h4 --max-age 4
  run_candidate 0 fixed_h8 --max-age 8
  run_candidate 0 fixed_h16 --max-age 16
) & fixed0=$!
(
  run_candidate 1 fixed_h2 --max-age 2
  run_candidate 1 fixed_h6 --max-age 6
  run_candidate 1 fixed_h12 --max-age 12
  run_candidate 1 continuous
) & fixed1=$!
wait "$fixed0"
wait "$fixed1"

PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" "$RUNNER" --mode derive \
  --selection-root "$VALIDATION_ROOT" --output "$RESULT_ROOT/reset_policy/derived_thresholds.json"

read -r accum50 accum75 accum90 disagreement support fb activation update < <(
  "$ARGOS_PYTHON" -c 'import json,sys; x=json.load(open(sys.argv[1])); a=x["accumulated_update"]; e=x["conservative_evidence"]; print(a["0.5"],a["0.75"],a["0.9"],e["disagreement_max"],e["warp_support_min"],e["fb_confidence_min"],e["temporal_activation_max"],e["update_magnitude_max"])' \
  "$RESULT_ROOT/reset_policy/derived_thresholds.json"
)

(
  run_candidate 0 accum_q50 --accumulated-update-max "$accum50"
  run_candidate 0 accum_q75 --accumulated-update-max "$accum75"
  run_candidate 0 accum_q90 --accumulated-update-max "$accum90"
) & adaptive0=$!
(
  common=(--disagreement-max "$disagreement" --warp-support-min "$support" --fb-confidence-min "$fb" --temporal-activation-max "$activation" --update-magnitude-max "$update")
  run_candidate 1 evidence_conservative "${common[@]}"
  run_candidate 1 hybrid_h4 --max-age 4 "${common[@]}"
  run_candidate 1 hybrid_h6 --max-age 6 "${common[@]}"
  run_candidate 1 hybrid_h8 --max-age 8 "${common[@]}"
) & adaptive1=$!
wait "$adaptive0"
wait "$adaptive1"

run_hard() {
  local gpu=$1
  local threshold=$2
  local name="threshold_${threshold}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" "$RUNNER" \
    --mode evaluate --split validation --device cuda:0 --workers 20 --preload-workers 20 \
    --policy-name hard_h4 --max-age 4 --hard-threshold "$threshold" \
    --output "$HARD_ROOT/$name" >"$HARD_ROOT/${name}.log" 2>&1
}
(
  run_hard 0 0.20
  run_hard 0 0.50
  run_hard 0 0.80
) & hard0=$!
(
  run_hard 1 0.35
  run_hard 1 0.65
) & hard1=$!
wait "$hard0"
wait "$hard1"

PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" "$RUNNER" --mode select \
  --selection-root "$VALIDATION_ROOT" --output "$RESULT_ROOT/reset_policy/final_reset_policy.json"
PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" "$RUNNER" --mode select \
  --selection-root "$HARD_ROOT" --output "$RESULT_ROOT/ablations/hard_endpoint/final_hard_policy.json"
mkdir -p "$RESULT_ROOT/reset_policy/h4_only/fixed_h4"
cp "$VALIDATION_ROOT/fixed_h4/summary.json" "$RESULT_ROOT/reset_policy/h4_only/fixed_h4/summary.json"
PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" "$RUNNER" --mode select \
  --selection-root "$RESULT_ROOT/reset_policy/h4_only" --output "$RESULT_ROOT/reset_policy/fixed_h4_policy.json"
date -Is >"$RESULT_ROOT/validation_policy_freeze.timestamp"
