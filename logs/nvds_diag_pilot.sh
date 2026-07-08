#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
python scripts/temporal_refinement/nvds_lite_causal/run_matrix.py \
  --configs A D --seeds 0 --steps 600 --eval-every 300 \
  --out results/03_temporal_refinement/nvds_lite_causal_pilot/diag_pilot 2>&1 | stdbuf -oL grep --line-buffered -viE 'warn|future'
echo NVDS_DIAG_DONE
