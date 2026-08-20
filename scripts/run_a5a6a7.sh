#!/bin/bash
# A5, A6 and A7 at seed 0: the three ablations of the SHIPPED head the reviewers asked for.
#
#   A5  trains without the forward-backward confidence cue, which is the only thing the
#       reverse SEA-RAFT pass (13.1 of 28.9 ms) exists to compute.
#   A6  drops the 25 quarter-resolution correlations, two thirds of the evidence space,
#       which the 142 -> 78 -> 38 sequence never tested.
#   A7  halves the head width, 154,874 -> 70,994 parameters, so "compact" becomes a
#       finding rather than a description.
#
# Pre-registered 2026-08-20 against the 38-channel head at three seeds, held-out Bad1,
# 0.37-point threshold. Any variant that clears the threshold on this run gets seeds 1
# and 2 before it is interpreted.
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
    for V in A5_no_fb_cue A6_geometry_only A7_half_width; do
        RUN="$ROOT/model_design/training_runs/ablation_$V"
        EPOCHS_DONE=$(( $(wc -l < "$RUN/training_history.csv" 2>/dev/null || echo 1) - 1 ))
        if [ "$EPOCHS_DONE" -ge 12 ]; then echo "=== $V already trained ($EPOCHS_DONE epochs)"; continue; fi
        [ "$EPOCHS_DONE" -gt 0 ] && echo "=== $V resuming from epoch $EPOCHS_DONE"
        echo "=== $V"
        "$PY" model_design/train_ablation.py --variant "$V" --workers 20 --preload-workers 20 \
            || echo "FAILED $V"
    done
    echo "A5A6A7 DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_a567 "$SELF --node $*"
