#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
python scripts/temporal_refinement/vdpp_style_causal/train_vdpp_causal.py \
  --temporal-input-mode shuffled_history --loss-mode spatial_plus_tgm --smoke \
  --out /dtu/p1/leopam/ARGOS/logs/vdpp_csmoke_out 2>&1 | grep -viE 'warn|future' | tail -4
echo "CSMOKE_DONE"
