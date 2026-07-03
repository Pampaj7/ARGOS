#!/bin/bash
export ESUB_BYPASS=1 ESUB_QUIET=1
bsub -q p1i -app h100app -gpu "num=1:mode=shared" -n 4 -R "span[hosts=1]" -J v32c_smoke -I bash -c '
set -euo pipefail
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4
source .miniconda/etc/profile.d/conda.sh
conda activate argos
python scripts/temporal_refinement/train_tiny_refiner_v3_2_hybrid_oracle.py \
  --output-root results/03_temporal_refinement/training/tiny_refiner_v3_2c_smoke \
  --base-checkpoint results/03_temporal_refinement/training/tiny_refiner_v3_2b_hybrid_oracle_freeze_detector/checkpoints/best.pt \
  --freeze-detector true --sparsity-weight 0.02 --residual-lr 3e-4 \
  --lr-schedule cosine --lr-decay-start-epoch 3 --epochs 5 \
  --max-frames 40 --crops-per-epoch 4096 --batch-size 512 \
  --num-workers 3 --early-stop-patience 5 --overwrite
echo TRAIN_EXIT=$?
'
echo BSUB_EXIT=$?
