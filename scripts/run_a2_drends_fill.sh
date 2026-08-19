#!/bin/bash
# S2M2-S and StereoAnywhere on DRENDS for the A2 head.
#
# The canonical head has all five estimators out of domain; A2 has three, so the transfer
# grid cannot be rebuilt around A2 without these two. They are the concrete blocker to
# re-canonicalising the paper, not a nice-to-have: a promoted method whose out-of-domain
# column is missing the two backbones the canonical column has would be exactly the
# unmatched comparison this project has already got wrong twice.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/drends_a2/seed0
RECS="Vid10_Liver_Med Vid11_Liver_High Vid12_Pancreas_Ext Vid13_Pancreas_Med Vid14_Pancreas_High"
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    for B in ${BACKBONES:-S2M2-S StereoAnywhere}; do
        if [ -f "$OUT/$B/summary.json" ]; then echo "=== $B already complete"; continue; fi
        [ -d "$OUT/$B" ] && mv "$OUT/$B" "$OUT/$B.incomplete-$(date +%s)"
        echo "=== $B"
        "$PY" -m model_design.comparison.drends_backbone_transfer --device cuda:0 \
            --backbone "$B" --module "model_design.comparison.ablation_h4:factory_a2" \
            --recordings $RECS --output "$OUT/$B" || echo "FAILED $B"
    done
    echo "A2 DRENDS FILL DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=10GB]" \
     -gpu "num=1:mode=shared" -env "all, BACKBONES=${BACKBONES:-S2M2-S StereoAnywhere}" \
     -J argos_a2fill "$SELF --node $*"
