#!/usr/bin/env bash
# Strict ARGOS v2 CODD-style Phase-1 campaign.  Dataset 7 is prohibited until
# all three seed checkpoints have completed validation-only selection.
set -euo pipefail
ROOT=/dtu/p1/leopam/ARGOS/ARGOS-V2
PYTHON=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
OUT="$ROOT/results/codd_style_fusion_probe/bida_memory_phase1/full_phase1"
cd "$ROOT"; mkdir -p "$OUT/seed_0" "$OUT/seed_1" "$OUT/seed_2"

COMMON=(scripts/run_codd_style_fusion_probe.py --workers 20 --preload-workers 20 --batch-size 4 --epochs 12 --learning-rate 0.0002 --weight-decay 0.0001 --clip-length 4 --coverage-threshold .50)
train() { mkdir -p "$OUT/seed_$1"; CUDA_VISIBLE_DEVICES="$2" PYTHONPATH="$ROOT" "$PYTHON" "${COMMON[@]}" --mode train --output "$OUT/seed_$1" --device cuda:0 --seed "$1" >"$OUT/seed_$1/train.log" 2>&1; }
evaluate() { mkdir -p "$OUT/seed_$1"; CUDA_VISIBLE_DEVICES="$2" PYTHONPATH="$ROOT" "$PYTHON" "${COMMON[@]}" --mode evaluate --output "$OUT/seed_$1" --device cuda:0 --seed "$1" >"$OUT/seed_$1/evaluate.log" 2>&1; }

train 0 0 & p0=$!
train 1 1 & p1=$!
wait "$p0"; wait "$p1"
train 2 0

# Frozen checkpoints only; these are the first commands permitted to open D7.
evaluate 0 0 & p0=$!
evaluate 1 1 & p1=$!
wait "$p0"; wait "$p1"
evaluate 2 0
PYTHONPATH="$ROOT" "$PYTHON" scripts/run_codd_style_fusion_probe.py --mode summarize --output "$OUT"
