#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
python scripts/temporal_refinement/nvds_lite_causal/run_matrix.py \
  --configs A --seeds 0 --batch 4 --steps 100 --eval-every 1000 \
  --out /dtu/p1/leopam/ARGOS/results/03_temporal_refinement/nvds_lite_causal_pilot/timing 2>&1 | stdbuf -oL grep --line-buffered -viE 'warn|future'
echo NVDS_TIMING_DONE
