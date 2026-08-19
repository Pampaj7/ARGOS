#!/bin/bash
# What the forward-backward confidence cue is worth in accuracy.
#
# It costs a whole SEA-RAFT pass, 13.7 of the module's 33.5 ms. The paper can only propose
# dropping it if it says what that costs, so the cue is replaced by a constant at inference
# and scored against the canonical row on the same fifteen cells.
#
# Two constants because the answer should not depend on which one: 1.0 says "always
# reliable", 0.5 says "always half". If they disagree the head is reading the level rather
# than the variation, which is itself worth knowing.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    cd "$ROOT" || exit 1
    for C in ${CONSTANTS:-1.0 0.5}; do
        DEST=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/definitive_evaluation/experimental_closure/d2_nofb_c$C
        if [ -f "$DEST/summary.csv" ]; then echo "=== c=$C already done"; continue; fi
        [ -d "$DEST" ] && mv "$DEST" "${DEST}.incomplete-$(date +%s)"
        echo "=== constant $C"
        "$PY" -m model_design.comparison.no_fb_sweep --constant "$C" --device cuda:0 || echo "FAILED c=$C"
    done
    echo "NO FB DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=10GB]" \
     -gpu "num=1:mode=shared" -env "all, CONSTANTS=${CONSTANTS:-1.0 0.5}" -J argos_nofb "$SELF --node $*"
