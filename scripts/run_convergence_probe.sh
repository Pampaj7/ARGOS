#!/bin/bash
# Twenty-four epochs of the shipped model against a locked budget of twelve.
#
# The paper claims the budget does not bind, and argues it from the epoch at which selection
# fires: 8, 11 and 11. A reviewer reads two of three seeds selecting at epoch 11 of 12 as an
# undertrained model, and is entitled to. The curve supports the claim better than that
# phrasing does -- from epoch 5 the epoch-to-epoch wobble is as large as the remaining gain,
# so the minimum lands by noise -- but that is an argument about a curve, and only a longer
# curve settles it.
#
# Nothing here is eligible for selection. If it keeps improving to 24, that is a limitation
# to report, not a model to adopt.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    "$PY" -m model_design.train_convergence_probe --epochs "${EPOCHS:-24}" \
          --workers 12 --preload-workers 12 || { echo "PROBE FAILED"; exit 1; }
    echo "CONVERGENCE PROBE DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=1:mode=shared" -env "all, EPOCHS=${EPOCHS:-24}" -J argos_converge "$SELF --node $*"
