#!/usr/bin/env bash
set -euo pipefail
cd /dtu/p1/leopam/ARGOS/ARGOS-V2
mkdir -p /tmp/leopam_argos_v2 results/raw_multi_anchor_temporal_refiner/logs
export TMPDIR=/tmp/leopam_argos_v2
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python scripts/run_raw_multi_anchor_temporal_refiner.py \
  --mode train --configuration cs1 --output results/raw_multi_anchor_temporal_refiner \
  --device cuda:0 --flow-batch-size 32 --batch-size 8 --workers 16 \
  --channels 32 --blocks 3 --epochs 10
