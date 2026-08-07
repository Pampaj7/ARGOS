#!/usr/bin/env bash
# Continue ARGOS v2 ablation execution after the first strict overfit gate.
set -euo pipefail

ARGOS_ROOT=/dtu/p1/leopam/ARGOS/ARGOS-V2
ARGOS_PYTHON=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
RESULT_ROOT="$ARGOS_ROOT/results/codd_style_bounded_memory_validation"
SMOKE_ROOT="$RESULT_ROOT/_temporary_smoke"
ABLATION_ROOT="$RESULT_ROOT/ablations"
cd "$ARGOS_ROOT"

# GPU1 can begin the already-passed no-feature full run while GPU0 completes
# the stricter no-recurrence overfit check.
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" scripts/run_codd_style_fusion_probe.py \
  --mode train --seed 0 --device cuda:0 --workers 20 --preload-workers 20 \
  --output "$ABLATION_ROOT/no_learned_stereo_evidence" --disable-learned-stereo-evidence \
  >"$ABLATION_ROOT/no_learned_stereo_evidence_train.log" 2>&1 & pid_feature=$!

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" scripts/run_codd_style_fusion_probe.py \
  --mode overfit --overfit-epochs 160 --seed 0 --device cuda:0 --workers 20 --preload-workers 20 \
  --output "$SMOKE_ROOT/no_recurrence_extended" --memory-state raw_previous \
  >"$SMOKE_ROOT/no_recurrence_extended_overfit.log" 2>&1
test -f "$SMOKE_ROOT/no_recurrence_extended/overfit_summary.json"
rm -rf "$SMOKE_ROOT/no_recurrence_extended"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ARGOS_ROOT" "$ARGOS_PYTHON" scripts/run_codd_style_fusion_probe.py \
  --mode train --seed 0 --device cuda:0 --workers 20 --preload-workers 20 \
  --output "$ABLATION_ROOT/no_recurrence" --memory-state raw_previous \
  >"$ABLATION_ROOT/no_recurrence_train.log" 2>&1 & pid_recurrence=$!

wait "$pid_feature"
wait "$pid_recurrence"
date -Is >"$RESULT_ROOT/ablation_training_complete.timestamp"
