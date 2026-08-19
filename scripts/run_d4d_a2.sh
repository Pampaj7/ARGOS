#!/bin/bash
# The no-reference control on the head the paper ships.
#
# The limitations section reports that TETHER reduces the D4D no-reference consistency
# score by 53.9% while copy-previous removes 97.3%, which is the argument that the metric
# is gameable. That 53.9% was measured on the 142-channel configuration -- the ablation --
# and reported in the paper's voice as "the module". The degenerate bounds are model-free
# and stay as they are; only the module's own arm is re-run.
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
    "$PY" -m model_design.comparison.run_comparison --dataset d4d \
        --module model_design.comparison.ablation_h4:factory_a2 --device cuda:0 \
        --output /dtu/p1/leopam/ARGOS/ARGOS_hand/results/d4d_degenerate/a2_factory
    echo "D4D A2 DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_d4da2 "$SELF --node $*"
