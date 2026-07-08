#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
python scripts/temporal_refinement/adaptation/temporal_eval_d4d.py \
  --clips-per-specimen 2 --max-frames 120 \
  --out results/03_temporal_refinement/adaptation/d4d_temporal_eval 2>/dev/null
echo "FULL_TEMPORAL_DONE rc=$?"
