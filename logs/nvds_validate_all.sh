#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
set -e
echo "=== STEP 1: rebuild aux cache (backward warp flow) ==="
python scripts/temporal_refinement/nvds_lite_causal/build_aux_cache.py 2>&1 | grep -viE 'warn|future'
echo "=== STEP 2: causal-leakage + gradient/graph validation ==="
python scripts/temporal_refinement/nvds_lite_causal/validate_causal_and_grad.py 2>&1 | grep -viE 'warn|future'
echo "=== STEP 3: 150-step A/B pilot (departure from identity, warp loss trajectory) ==="
for CFG in A B; do
  echo "--- pilot $CFG ---"
  python scripts/temporal_refinement/nvds_lite_causal/train_nvds_lite.py --config $CFG --seed 0 \
    --steps 150 --eval-every 50 --out /tmp/nvds_pilot_ab 2>&1 | grep -viE 'warn|future'
done
echo NVDS_VALIDATE_ALL_DONE
