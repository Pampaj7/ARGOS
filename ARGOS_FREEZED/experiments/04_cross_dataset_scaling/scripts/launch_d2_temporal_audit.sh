#!/usr/bin/env bash
set -euo pipefail
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_FREEZED
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
cd "$ROOT"
"$PY" scripts/verify_freeze.py
"$PY" experiments/04_cross_dataset_scaling/scripts/run_d2_temporal_audit.py --dataset-id 2 "$@"
