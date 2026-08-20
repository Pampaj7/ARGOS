#!/bin/bash
# Motion-compensated temporal fidelity on the head the paper ships.
#
# The main paper says of the TEPE_corr / OPW family: "it improves on all five D7 backbones
# at all four horizons but the two measures disagree on D2". That is a claim about our
# method, and it was measured on the 142-channel ablation, because the script's default
# module is canonical_h4_masked and nobody passed --module when the head changed.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/temporal_corrected_a2
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
    mkdir -p "$OUT"
    for SPLIT in d2 d7; do
        echo "=== $SPLIT"
        "$PY" scripts/evaluate_temporal_corrected.py --split "$SPLIT" \
            --module model_design.comparison.ablation_h4:factory_a2 \
            --device cuda:0 --output "$OUT" || echo "FAILED $SPLIT"
    done
    echo "TEMPORAL CORRECTED A2 DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_tcorr "$SELF --node $*"
