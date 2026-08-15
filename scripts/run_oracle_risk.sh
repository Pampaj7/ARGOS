#!/bin/bash
# Dispatch the oracle/selective-risk analysis to the p1i H100 node (720 min limit).
#   setsid nohup scripts/run_oracle_risk.sh [args...] > logs/oracle_risk.log 2>&1 < /dev/null &
# Picks whichever card has free memory: the queue only grants shared GPUs without a wait.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
EVAL=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4/scripts/analyze_oracle_and_risk.py
SELF=$(readlink -f "$0")

if [ "${1:-}" = "--node" ]; then
    shift
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                           | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES
    echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES free_mib=$(nvidia-smi --query-gpu=memory.free \
          --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")"
    exec "$PY" "$EVAL" --device cuda:0 "$@"
fi

export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 \
     -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=2:mode=shared" \
     -J argos_oracle_risk \
     "$SELF --node $*"
