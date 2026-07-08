#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
python scripts/temporal_refinement/vdpp_style_causal/train_vdpp_causal.py --mode tgm --smoke \
  --out /dtu/p1/leopam/ARGOS/logs/vdpp_smoke_out 2>&1 | grep -viE 'warn|future'
echo "VDPP_SMOKE_DONE rc=$?"
