#!/bin/bash
# Which branch of w = r * f produces the gain, the review's second question.
#
# The paper reports and analyses the product and never separates the factors. Both runs
# reuse the trained head unchanged and only suppress one factor at inference, so the
# difference is the branch and not a different model.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    # Prefer a GPU with no compute process at all, and fall back to most-free-memory. Free
    # memory alone is a poor criterion when jobs dispatch minutes apart: they read the same
    # snapshot and pile onto the same device while the other sits at 0% -- observed twice.
    BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v busy="$BUSY" '{used=index(busy,$2)>0; print used, -$3, $1}' \
        | sort -k1,1n -k2,2n | head -1 | awk '{print $3}')
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    cd "$ROOT" || exit 1
    for B in ${BRANCHES:-reset_only fusion_only}; do
        DEST=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/definitive_evaluation/experimental_closure/d2_branch_$B
        if [ -f "$DEST/summary.csv" ]; then echo "=== $B already done"; continue; fi
        [ -d "$DEST" ] && mv "$DEST" "${DEST}.incomplete-$(date +%s)"
        echo "=== $B"
        "$PY" -m model_design.comparison.branch_sweep --branch "$B" --device cuda:0 || echo "FAILED $B"
    done
    echo "BRANCH ABLATION DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=10GB]" \
     -gpu "num=1:mode=shared" -env "all, BRANCHES=${BRANCHES:-reset_only fusion_only}" \
     -J argos_branch "$SELF --node $*"
