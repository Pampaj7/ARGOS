#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/dtu/p1/leopam/ARGOS/ARGOS-V2:/dtu/p1/leopam/ARGOS/ARGOS-V2/scripts
cd /dtu/p1/leopam/ARGOS/ARGOS-V2
mkdir -p results/hybrid_temporal_memory_oracle_audit
export PYTHONUNBUFFERED=1
exec /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  scripts/run_hybrid_temporal_memory_oracle_audit.py \
  --mode evaluate --stage validation \
  --output results/hybrid_temporal_memory_oracle_audit \
  --sequences dataset_2_keyframe_2 dataset_2_keyframe_3 \
  --device cuda:0 --batch-size 1 --workers 32
