#!/bin/bash
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
    cd "$ROOT" || exit 1
    for SPEC in degenerate_policies:factory degenerate_policies:factory_warped canonical_h4_masked:factory; do
        NAME=$(echo "$SPEC" | tr ':.' '__')
        echo "=== $SPEC"
        "$PY" model_design/comparison/run_comparison.py --dataset d4d \
            --module "model_design.comparison.$SPEC" --device cuda:0 \
            --output /dtu/p1/leopam/ARGOS/ARGOS_hand/results/d4d_degenerate/"$NAME" "$@" || echo "FAILED $SPEC"
    done
    echo "D4D DEGENERATE DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" -gpu "num=2:mode=shared" -J argos_d4ddeg "$SELF --node $*"
