#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos; cd /dtu/p1/leopam/ARGOS; export PYTHONPATH="$(pwd)"
python scripts/temporal_refinement/vdpp_style_causal/train_vdpp_causal.py --temporal-input-mode shuffled_history --loss-mode spatial_plus_tgm --seed 1 --lam-tgm 1.0 --clip-len 8 --steps 700 --eval-every 700 --out results/03_temporal_refinement/vdpp_style_causal_confirmation/runs 2>&1 | grep -viE 'warn|future'|tail -1
echo "SHUF1_DONE"
