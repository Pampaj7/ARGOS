#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
python scripts/temporal_refinement/nvds_lite_causal/validate_flow.py 2>&1 | grep -viE 'warn|future'
echo NVDS_VALFLOW_DONE
