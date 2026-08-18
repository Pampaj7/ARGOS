#!/bin/bash
# S2M2-S and StereoAnywhere on DRENDS, completing the backbone x domain grid.
#
# DRENDS was run with three of the five estimators, so the transfer table had two holes,
# both on the "seen backbone" side -- which left the out-of-domain column reading one seen
# against two unseen. These two runs close it, and the merged results table then carries
# every backbone in every arena with no missing cell to explain.
#
# Same module, checkpoint and protocol as results/drends_masked: canonical_h4_masked, one
# frozen checkpoint, nothing trained or tuned. The evaluator has changed once since those
# runs -- the default recording list was fixed after it silently produced a one-sequence
# result -- so the recordings are passed explicitly here and the metric code is untouched
# between the two halves of the table.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/drends_masked
RECS="Vid10_Liver_Med Vid11_Liver_High Vid12_Pancreas_Ext Vid13_Pancreas_Med Vid14_Pancreas_High"
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    for B in S2M2-S StereoAnywhere; do
        # A finished run is one that wrote its summary, not one that made a directory.
        if [ -f "$OUT/$B/summary.json" ]; then echo "=== $B already complete"; continue; fi
        echo "=== $B"
        "$PY" -m model_design.comparison.drends_backbone_transfer --device cuda:0 \
            --backbone "$B" --module "model_design.comparison.canonical_h4_masked:factory" \
            --recordings $RECS --output "$OUT/$B" || echo "FAILED $B"
    done
    echo "DRENDS MISSING BACKBONES DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=1:mode=shared" -J argos_drendsfill "$SELF --node $*"
