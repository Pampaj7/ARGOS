#!/bin/bash
export ESUB_BYPASS=1 ESUB_QUIET=1
bsub -q p1i -app h100app -gpu "num=1:mode=shared" -n 4 -R "span[hosts=1]" -J v33b_smoke -I bash -c '
set -euo pipefail
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4
source .miniconda/etc/profile.d/conda.sh
conda activate argos
python scripts/temporal_refinement/train_tiny_refiner_v3_3b_hard_negative.py \
  --output-root results/03_temporal_refinement/training/tiny_refiner_v3_3b_hard_negative_smoke \
  --max-frames 40 --crops-per-epoch 4096 --batch-size 512 --epochs 2 \
  --num-workers 3 --early-stop-patience 2 --overwrite
echo TRAIN_EXIT=$?
'
echo BSUB_EXIT=$?
