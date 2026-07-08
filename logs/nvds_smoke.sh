#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
for CFG in A B C D E F; do
  echo "=== CONFIG $CFG ==="
  python scripts/temporal_refinement/nvds_lite_causal/train_nvds_lite.py --config $CFG --smoke --device cuda --out /tmp/nvds_smoke_runs 2>&1 | grep -viE 'warn|future'
done
echo NVDS_SMOKE_DONE
