#!/bin/bash
# The two-parameter unlearned rule the review says the closure is missing.
#
# w = alpha * exp(-|raw - aligned| / tau): spatially varying, no learning, driven by the
# residual the paper's own analysis says the head responds to. A sweep on D2 only -- the
# development split, which is where the learned policy was also selected, so the comparison
# gives the unlearned rule the same tuning freedom the head had.
#
# If the best (alpha, tau) matches the learned head, 177k parameters buy a tuned constant.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/residual_policy
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    cd "$ROOT" || exit 1
    # One alpha per job: sixteen points at twenty minutes each is six hours in series and
    # ninety minutes across four jobs, and the points are independent. Each job owns its own
    # output directories, so parallel runs cannot race on the same one.
    for ALPHA in ${ALPHAS:-0.3 0.5 0.8 1.0}; do
        for TAU in 0.25 0.5 1.0 2.0; do
            DEST="$OUT/a${ALPHA}_t${TAU}"
            if [ -f "$DEST/summary.csv" ]; then echo "=== a=$ALPHA t=$TAU already done"; continue; fi
            [ -d "$DEST" ] && mv "$DEST" "${DEST}.incomplete-$(date +%s)"
            echo "=== alpha=$ALPHA tau=$TAU"
            "$PY" -m model_design.comparison.residual_sweep --alpha "$ALPHA" --tau "$TAU" \
                --device cuda:0 --output "$DEST" || echo "FAILED a=$ALPHA t=$TAU"
        done
    done
    echo "RESIDUAL SWEEP DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=10GB]" \
     -gpu "num=1:mode=shared" -env "all, ALPHAS=${ALPHAS:-0.3 0.5 0.8 1.0}" \
     -J "argos_residual" "$SELF --node $*"
