#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/dtu/p1/leopam/ARGOS/ARGOS-V2:/dtu/p1/leopam/ARGOS/ARGOS-V2/scripts
cd /dtu/p1/leopam/ARGOS/ARGOS-V2
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  scripts/run_hybrid_temporal_memory_oracle_audit.py \
  --mode smoke \
  --output /tmp/argos_v2_hybrid_memory_smoke \
  --device cuda:0 \
  --batch-size 1 \
  --workers 4 \
  --max-frames 10
