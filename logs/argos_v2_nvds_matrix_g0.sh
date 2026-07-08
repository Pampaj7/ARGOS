#!/usr/bin/env bash
set -euo pipefail
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python scripts/temporal_refinement/nvds_lite_causal/run_matrix.py \
  --configs A B C \
  --seeds 0 1 2 \
  --batch 4 \
  --steps 1200 \
  --eval-every 1200 \
  --out results/03_temporal_refinement/nvds_lite_causal_pilot/runs_argos_v2 \
  --device cuda
