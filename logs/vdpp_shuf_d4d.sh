#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
python scripts/temporal_refinement/vdpp_style_causal/train_vdpp_causal.py \
  --mode shuffled --clip-len 8 --steps 1500 --seed 0 --eval-every 1500 \
  --out results/03_temporal_refinement/vdpp_style_causal_pilot/runs 2>&1 | grep -viE 'warn|future' | tail -2
echo "SHUFFLED_DONE"
python scripts/temporal_refinement/vdpp_style_causal/eval_vdpp_d4d.py \
  --ckpt results/03_temporal_refinement/vdpp_style_causal_pilot/runs/tgm__clip8__seed0/best.pt \
  --clips-per-specimen 2 --max-frames 120 2>&1 | grep -viE 'warn|future' | tail -25
echo "VDPP_D4D_DONE"
