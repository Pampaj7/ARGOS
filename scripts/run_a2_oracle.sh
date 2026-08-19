#!/bin/bash
# Recompute the raw-versus-memory oracle on the head the paper ships.
#
# The closure table reports a 12.30% oracle ceiling and says TETHER recovers 30.0% of it.
# That ceiling was computed from the 142-channel ablation's canonical_h4 bundles: the
# oracle selects, with ground truth, between raw and the aligned memory, and under H=4 the
# memory is the previous FUSED output on three frames out of four. A different head gives a
# different memory and therefore a different ceiling, so the published ratio divides
# TETHER's gain by the ablation's ceiling.
#
# One method, fifteen cells, roughly the 75 minutes the h4 half of the h4h6 shard took.
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
    "$PY" -m model_design.comparison.promoted_head_closure \
        --head A2_no_learned_evidence --methods canonical_h4 --shard oracle --device cuda:0
    echo "A2 ORACLE DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_a2oracle "$SELF --node $*"
