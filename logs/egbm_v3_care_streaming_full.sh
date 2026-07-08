#!/bin/bash
export ESUB_BYPASS=1 ESUB_QUIET=1
bsub -q p1i -app h100app -gpu "num=1:mode=shared" -n 4 -R "span[hosts=1]" -J v3s_full -I bash -c '
set -euo pipefail
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
source .miniconda/etc/profile.d/conda.sh
conda activate argos
echo NODE=$(hostname)
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python scripts/temporal_refinement/train_egbm_v3_care_streaming.py \
  --output-root results/03_temporal_refinement/training/egbm_v3_care_streaming \
  --batch-size 12 --chunk-length 16 --crops-per-epoch 8000 \
  --stage1-epochs 8 --stage2-epochs 16 --stage3-epochs 24 \
  --early-stop-patience 8 \
  --num-workers 20 --fresh
echo TRAIN_EXIT=$?
python scripts/temporal_refinement/eval_scripts/benchmark_egbm_v3_care_streaming.py \
  --output-root results/03_temporal_refinement/training/egbm_v3_care_streaming --batch 16 || true
echo BENCH_EXIT=$?
'
echo BSUB_EXIT=$?
