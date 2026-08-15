#!/bin/bash
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
EVAL=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4/scripts/measure_runtime.py
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES
    echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")"
    exec "$PY" "$EVAL" --device cuda:0 "$@"
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" -gpu "num=2:mode=shared" -J argos_runtime "$SELF --node $*"
