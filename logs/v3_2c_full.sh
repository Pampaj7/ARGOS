#!/bin/bash
export ESUB_BYPASS=1 ESUB_QUIET=1
bsub -q p1i -app h100app -gpu "num=2:mode=shared" -n 4 -R "span[hosts=1]" -J v32c_full -I bash -c '
set -euo pipefail
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4
source .miniconda/etc/profile.d/conda.sh
conda activate argos
echo "NODE=$(hostname)"
python scripts/temporal_refinement/train_tiny_refiner_v3_2_hybrid_oracle.py \
  --output-root results/03_temporal_refinement/training/tiny_refiner_v3_2c_hybrid_oracle_freeze_detector_long \
  --base-checkpoint results/03_temporal_refinement/training/tiny_refiner_v3_2b_hybrid_oracle_freeze_detector/checkpoints/best.pt \
  --freeze-detector true --sparsity-weight 0.02 --residual-lr 3e-4 \
  --lr-schedule cosine --lr-decay-start-epoch 16 \
  --epochs 32 --early-stop-patience 8 \
  --baseline-selected-mae 11.0075 --baseline-oracle-gap-recovered 0.0699 --baseline-selected-new-bad3 4.79 \
  --num-workers 24 --overwrite
echo TRAIN_EXIT=$?
'
echo BSUB_EXIT=$?
