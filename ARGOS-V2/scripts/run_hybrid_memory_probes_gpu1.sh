#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/dtu/p1/leopam/ARGOS/ARGOS-V2:/dtu/p1/leopam/ARGOS/ARGOS-V2/scripts
export PYTHONUNBUFFERED=1
cd /dtu/p1/leopam/ARGOS/ARGOS-V2
exec /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  scripts/run_hybrid_temporal_memory_oracle_audit.py \
  --mode probes \
  --output results/hybrid_temporal_memory_oracle_audit \
  --device cuda:0 --batch-size 1 --workers 32 \
  --max-frames 64 --probe-pixels-per-frame 64
