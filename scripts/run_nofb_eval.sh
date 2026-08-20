#!/bin/bash
# The shipped checkpoint with C^FB pinned at inference, on both grids (declaration JSON
# beside the output). Decides whether "train with the cue, deploy without" survives D7.
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
    "$PY" model_design/comparison/run_definitive_evaluation.py --datasets scared-d2 scared-d7 \
        --module "model_design.comparison.no_fb_cue:factory" --device cuda:0 \
        --output /dtu/p1/leopam/ARGOS/ARGOS_hand/results/ablation_eval/nofb_inference \
        || echo "FAILED nofb"
    echo "NOFB EVAL DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_nofb "$SELF --node $*"
