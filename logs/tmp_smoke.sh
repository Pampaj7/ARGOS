#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
python scripts/temporal_refinement/adaptation/temporal_eval_d4d.py --smoke \
  --out results/03_temporal_refinement/adaptation/d4d_temporal_eval_smoke
