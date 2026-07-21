#!/usr/bin/env bash
# Controlled ARGOS v2 information probe.  No model, loss or decision-policy
# choice is made here: this launcher only reproduces the validated strict 128x8
# campaign with the predeclared full census-cost evidence.
set -euo pipefail

ROOT=/dtu/p1/leopam/ARGOS/ARGOS-V2
PYTHON=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
OUT="$ROOT/results/utility_selector_information_probe/candidate_conditioned_stereo_evidence_128x8/full_128x8"
cd "$ROOT"
mkdir -p "$OUT"

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

train_seed() {
  local seed=$1 gpu=$2
  mkdir -p "$OUT/seed_${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PYTHON" "${COMMON[@]}" --mode train \
    --output "$OUT/seed_${seed}" --device cuda:0 --seed "$seed" \
    >"$OUT/seed_${seed}/train.log" 2>&1
}

calibrate_seed() {
  local seed=$1 gpu=$2
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PYTHON" "${COMMON[@]}" --mode calibrate \
    --output "$OUT/seed_${seed}" --device cuda:0 --seed "$seed" \
    >"$OUT/seed_${seed}/calibrate.log" 2>&1
}

evaluate_seed() {
  local seed=$1 gpu=$2
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PYTHON" "${COMMON[@]}" --mode evaluate \
    --output "$OUT/seed_${seed}" --device cuda:0 --seed "$seed" \
    >"$OUT/seed_${seed}/evaluate.log" 2>&1
}

# Dataset 7 is not touched until all three training runs and all validation-
# only calibration points are complete and frozen.
train_seed 0 0 & pid0=$!
train_seed 1 1 & pid1=$!
wait "$pid0"
wait "$pid1"
train_seed 2 0

calibrate_seed 0 0 & pid0=$!
calibrate_seed 1 1 & pid1=$!
wait "$pid0"
wait "$pid1"
calibrate_seed 2 0

evaluate_seed 0 0 & pid0=$!
evaluate_seed 1 1 & pid1=$!
wait "$pid0"
wait "$pid1"
evaluate_seed 2 0

PYTHONPATH="$ROOT" "$PYTHON" scripts/run_utility_memory_selector.py --mode summarize --output "$OUT"
