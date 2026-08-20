#!/bin/bash
# Evaluate A5 (forward-backward confidence pinned to 1.0) on the two SCARED-C grids.
#
# A5 decides a runtime claim, not just an ablation row: the reverse SEA-RAFT pass exists
# only to produce this channel, and costs 13.1 ms of the module's 28.9. If A5 holds the
# pre-registered endpoint, the paper may say the pass is droppable; if it degrades, the
# 45% of runtime buys accuracy and the claim dies. Either way the number has to exist.
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
    DEST=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/ablation_eval/a5_realfb
    "$PY" model_design/comparison/run_definitive_evaluation.py --datasets scared-d2 scared-d7 \
        --module "model_design.comparison.ablation_h4:factory_a5_realfb" --device cuda:0 \
        --output "$DEST" || echo "FAILED a5rfb"
    echo "A5 REALFB EVAL DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_a5rfb "$SELF --node $*"
