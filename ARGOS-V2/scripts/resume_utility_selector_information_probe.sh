#!/usr/bin/env bash
# Resume the interrupted strict information probe.  This intentionally changes
# no experimental setting: seed 2 resumes from last.pt, then all three seeds
# calibrate on dataset_2 before any dataset_7 evaluation is permitted.
set -euo pipefail

ROOT=/dtu/p1/leopam/ARGOS/ARGOS-V2
PYTHON=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
OUT="$ROOT/results/utility_selector_information_probe/candidate_conditioned_stereo_evidence_128x8/full_128x8"
cd "$ROOT"

COMMON=(
  scripts/run_utility_memory_selector.py
  --workers 20 --preload-workers 0 --batch-size 64 --epochs 12
  --channels 128 --blocks 8 --learning-rate 0.002 --weight-decay 0.0001
  --objective legacy --coverage-threshold 0.50 --epsilon 0.10
  --crop-height 96 --crop-width 120 --sampler hierarchical_dataset
  --stereo-matching-evidence full
  --train-sequences dataset_1_keyframe_2 dataset_1_keyframe_3
    dataset_3_keyframe_1 dataset_3_keyframe_2 dataset_3_keyframe_3 dataset_3_keyframe_4
    dataset_6_keyframe_1 dataset_6_keyframe_2 dataset_6_keyframe_3 dataset_6_keyframe_4
  --validation-sequences dataset_2_keyframe_2 dataset_2_keyframe_3 dataset_2_keyframe_4
  --test-sequences dataset_7_keyframe_1 dataset_7_keyframe_2 dataset_7_keyframe_3 dataset_7_keyframe_4
  --strict-dataset-id-disjoint
)

run() {
  local mode=$1 seed=$2
  PYTHONPATH="$ROOT" "$PYTHON" "${COMMON[@]}" --mode "$mode" \
    --output "$OUT/seed_${seed}" --device cuda:0 --seed "$seed" \
    >"$OUT/seed_${seed}/${mode}_resume.log" 2>&1
}

# This uses the atomic epoch-5 last.pt already written before LSF termination.
run train 2

# Dataset 7 is still not opened before all validation-only calibration finishes.
run calibrate 0
run calibrate 1
run calibrate 2

run evaluate 0
run evaluate 1
run evaluate 2

PYTHONPATH="$ROOT" "$PYTHON" scripts/run_utility_memory_selector.py --mode summarize --output "$OUT"
