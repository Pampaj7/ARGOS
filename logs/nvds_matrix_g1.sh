#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
python scripts/temporal_refinement/nvds_lite_causal/run_matrix.py \
  --configs D E F --seeds 0 1 2 --batch 8 --steps 1200 --eval-every 1200 \
  --out results/03_temporal_refinement/nvds_lite_causal_pilot/runs 2>&1 | grep --line-buffered -viE 'warn|future'
echo NVDS_MATRIX_G1_DONE
