#!/bin/bash
# Dispatch the frozen-transfer evaluation to the p1i H100 node (720 min limit).
#   setsid nohup scripts/run_frozen_transfer_eval.sh [--open-d7] > logs/frozen_transfer_eval.log 2>&1 < /dev/null &
#
# The queue only hands out shared GPUs without a wait, and the node is multi-tenant, so the
# job picks whichever card actually has free memory instead of trusting cuda:0.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
EVAL=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4/scripts/frozen_transfer_eval.py
SELF=$(readlink -f "$0")

if [ "${1:-}" = "--node" ]; then
    shift
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                           | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES
    echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES free_mib=$(nvidia-smi --query-gpu=memory.free \
          --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")"
    exec "$PY" "$EVAL" --device cuda:0 --flow-batch-size 32 "$@"
fi

export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 \
     -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=2:mode=shared" \
     -J argos_frozen_transfer \
     "$SELF --node $*"
