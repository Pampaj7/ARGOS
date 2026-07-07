#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
for MODE in tgm current_frame shuffled; do
  python scripts/temporal_refinement/vdpp_style_causal/train_vdpp_causal.py \
    --mode $MODE --clip-len 8 --steps 1500 --seed 0 --eval-every 1500 \
    --out results/03_temporal_refinement/vdpp_style_causal_pilot/runs 2>&1 | grep -viE 'warn|future' | tail -2
done
echo "VDPP_REST_DONE"
