#!/bin/bash
# Stage timing of A3 (full-resolution context), for the paper's cost argument (T2.2).
# The text declines A3 on "16x the context-branch compute"; a reviewer pairing that with
# the 1.00 ms head concludes we refused a better configuration to save ~3% of the module.
# This measures the branch in absolute ms so the sentence can be honest either way.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v busy="$BUSY" '{used=index(busy,$2)>0; print used, -$3, $1}' \
        | sort -k1,1n -k2,2n | head -1 | awk '{print $3}')
    echo "picked physical GPU $CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "$PY" scripts/measure_runtime.py --device cuda:0 \
        --module "model_design.comparison.ablation_h4:factory_a3" \
        || echo "FAILED a3 runtime"
    echo "A3 RUNTIME DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_a3rt "$SELF --node $*"
