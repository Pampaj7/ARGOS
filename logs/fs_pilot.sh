#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
for MODE in zero_shot calibration_only head_only full scratch; do
  SPLIT=4session_seed1
  if [ "$MODE" = "zero_shot" ]; then SPLIT=none; fi
  python scripts/temporal_refinement/adaptation/train_d4d_few_shot_adapter.py \
    --model EGBM-v3-CARE-S --adaptation-mode "$MODE" --split "$SPLIT" --seed 0 2>/dev/null | tail -14
done
