#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-results/03_temporal_refinement/training/egbm_v3_care_streaming}"
PY="${PY:-.miniconda/envs/argos/bin/python}"

mkdir -p logs
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PY" scripts/temporal_refinement/train_egbm_v3_care_streaming.py \
  --output-root "$OUT" \
  --stage1-epochs "${STAGE1_EPOCHS:-6}" \
  --stage2-epochs "${STAGE2_EPOCHS:-10}" \
  --stage3-epochs "${STAGE3_EPOCHS:-12}" \
  --crops-per-epoch "${CROPS_PER_EPOCH:-4096}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --chunk-length "${CHUNK_LENGTH:-16}" \
  --crop-size 96 \
  --eval-batch-size 32 \
  --eval-clip-batch 8 \
  --num-workers "${NUM_WORKERS:-24}" \
  --prefetch-factor "${PREFETCH_FACTOR:-4}" \
  --fresh true
echo "TRAIN_EXIT=0"

"$PY" scripts/temporal_refinement/eval_scripts/benchmark_egbm_v3_care_streaming.py \
  --output-root "$OUT" \
  --batch "${BENCH_BATCH:-16}"
echo "BENCH_EXIT=0"
