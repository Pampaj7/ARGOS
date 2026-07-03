#!/bin/bash
export ESUB_BYPASS=1 ESUB_QUIET=1
bsub -q p1i -app h100app -gpu "num=1:mode=shared" -n 4 -R "span[hosts=1]" -J v33_calib -I bash -c '
set -euo pipefail
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4
source .miniconda/etc/profile.d/conda.sh
conda activate argos
python scripts/temporal_refinement/eval_scripts/calibrate_v3_3_failure_mode_thresholds.py --num-workers 8
echo TRAIN_EXIT=$?
'
echo BSUB_EXIT=$?
