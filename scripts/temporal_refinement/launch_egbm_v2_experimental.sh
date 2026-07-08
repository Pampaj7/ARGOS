#!/usr/bin/env bash
set -u
cd /dtu/p1/leopam/ARGOS || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="results/03_temporal_refinement/training/egbm_v2_experimental"
PY=".miniconda/envs/argos/bin/python"

"$PY" scripts/temporal_refinement/train_egbm_v2_experimental.py \
  --output-root "$OUT" \
  --stage1-epochs 6 \
  --stage2-epochs 10 \
  --stage3-epochs 12 \
  --crops-per-epoch 60000 \
  --batch-size 128 \
  --eval-batch-size 48 \
  --eval-clip-batch 16 \
  --num-workers 24 \
  --prefetch-factor 4 \
  --overwrite true
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"

if [ "$TRAIN_EXIT" -eq 0 ]; then
  "$PY" scripts/temporal_refinement/eval_scripts/benchmark_egbm_v2.py \
    --egbm-root "$OUT" \
    --output-root "$OUT" \
    --eval-batch-size 64 \
    --num-workers 16 \
    --overwrite true
  EVAL_EXIT=$?
  echo "EVAL_EXIT=$EVAL_EXIT"
  exit "$EVAL_EXIT"
fi

exit "$TRAIN_EXIT"
