#!/bin/bash
# A hundred epochs of the shipped model against a locked budget of twelve.
#
# The paper claims the budget does not bind, and argues it from the epoch at which selection
# fires: 8, 11 and 11. A reviewer reads two of three seeds selecting at epoch 11 of 12 as an
# undertrained model, and is entitled to. The curve supports the claim better than that
# phrasing does -- from epoch 5 the epoch-to-epoch wobble is as large as the remaining gain,
# so the minimum lands by noise -- but that is an argument about a curve, and only a longer
# curve settles it.
#
# Eight times the budget, because doubling it would only move the question. Nothing here is
# eligible for selection: if it keeps improving, that is a limitation to report, not a model
# to adopt. About 14 min/epoch uncontended, so roughly a day.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    # num=2 above is what makes this work. With num=1 LSF grants one device and this
    # override can name the other, which CUDA then cannot see at all -- measured the hard
    # way: "No CUDA GPUs are available" from a pin that looked reasonable.
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    echo "picked physical GPU $CUDA_VISIBLE_DEVICES of: $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr '\n' ' ')"
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    "$PY" -m model_design.train_convergence_probe --epochs "${EPOCHS:-100}" \
          --workers 12 --preload-workers 12 || { echo "PROBE FAILED"; exit 1; }
    echo "CONVERGENCE PROBE DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -env "all, EPOCHS=${EPOCHS:-100}" -J argos_converge "$SELF --node $*"
