#!/bin/bash
export ESUB_BYPASS=1 ESUB_QUIET=1
bsub -q p1i -app h100app -gpu "num=1:mode=shared" -n 4 -R "span[hosts=1]" -J egbm_full -I bash -c '
set -euo pipefail
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source .miniconda/etc/profile.d/conda.sh
conda activate argos
echo "NODE=$(hostname)"
python scripts/temporal_refinement/train_experimental_refiner_vx.py \
  --batch-size 192 --eval-batch-size 32 --num-workers 24 --overwrite
echo TRAIN_EXIT=$?
'
echo BSUB_EXIT=$?
