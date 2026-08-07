#!/usr/bin/env bash
set -euo pipefail
ARGOS_ROOT=/dtu/p1/leopam/ARGOS/ARGOS-V2
ARGOS_PYTHON=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
RESULT_ROOT="$ARGOS_ROOT/results/codd_style_bounded_memory_validation"
SMOKE_DIR="$RESULT_ROOT/_temporary_policy_smoke_now"
cd "$ARGOS_ROOT"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" scripts/run_codd_style_bounded_memory_validation.py \
  --mode evaluate --split validation --tiny --device cuda:0 --workers 2 --preload-workers 2 \
  --policy-name smoke_h4 --max-age 4 --output "$SMOKE_DIR" >"$RESULT_ROOT/policy_smoke_now.log" 2>&1
test -f "$SMOKE_DIR/summary.json"
rm -rf "$SMOKE_DIR"
