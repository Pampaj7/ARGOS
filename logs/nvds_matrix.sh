#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
R=results/03_temporal_refinement/nvds_lite_causal_pilot
for CFG in A B C D E F; do
  for SEED in 0 1 2; do
    if ls $R/runs/${CFG}__*__seed${SEED}/config.json >/dev/null 2>&1; then echo "skip ${CFG} seed${SEED}"; continue; fi
    python scripts/temporal_refinement/nvds_lite_causal/train_nvds_lite.py --config $CFG --seed $SEED --clip-len 8 --steps 1200 --eval-every 1200 --out $R/runs 2>&1 | grep -viE 'warn|future' | tail -1
  done
done
echo NVDS_MATRIX_DONE
