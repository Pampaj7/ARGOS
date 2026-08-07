#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/dtu/p1/leopam/ARGOS/ARGOS-V2:/dtu/p1/leopam/ARGOS/ARGOS-V2/scripts
export PYTHONUNBUFFERED=1
cd /dtu/p1/leopam/ARGOS/ARGOS-V2
exec /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  scripts/run_hybrid_temporal_memory_oracle_audit.py \
  --mode evaluate --stage test \
  --output results/hybrid_temporal_memory_oracle_audit \
  --sequences dataset_7_keyframe_4 \
  --device cuda:0 --batch-size 1 --workers 32
