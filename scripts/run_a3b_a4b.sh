#!/bin/bash
# A3b and A4b at seed 0: the A3 and A4 deviations re-asked of the SHIPPED 38-channel head.
#
#   A3b  the context branch at full resolution instead of a quarter (SingleResolutionHead),
#        built on the 38-channel evidence space instead of the 142-channel one.
#   A4b  the zero-initialised additive escape from the convex interval (RelaxedConvexityHead),
#        likewise on the shipped head.
#
# The original A3 and A4 rows were measured against the 142-channel base the paper no
# longer ships, so their conclusions describe a different model. Pre-registered
# 2026-08-20 against the 38-channel head at three seeds, held-out D7 Bad1, 0.37-point
# threshold. Any variant that clears the threshold on this run gets seeds 1 and 2 before
# it is interpreted.
#
# Dispatches only once LSF job 29157969 (the A5/A6 training job) has ended -- ended(),
# not done(), because the watcher bkills that job after A6 and a bkill is EXIT, not DONE.
# The gate exists so a third concurrent training does not land on GPUs already carrying
# one training each; the evaluation jobs sharing the node are light and are handled by
# the GPU picker below at dispatch time. Note the pend is interactive (p1i requires
# bsub -I), so launch this under nohup with a log or the pend dies with the shell.
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
    for V in A3b_single_resolution_38ch A4b_relaxed_convexity_38ch; do
        RUN="$ROOT/model_design/training_runs/ablation_$V"
        EPOCHS_DONE=$(( $(wc -l < "$RUN/training_history.csv" 2>/dev/null || echo 1) - 1 ))
        if [ "$EPOCHS_DONE" -ge 12 ]; then echo "=== $V already trained ($EPOCHS_DONE epochs)"; continue; fi
        [ "$EPOCHS_DONE" -gt 0 ] && echo "=== $V resuming from epoch $EPOCHS_DONE"
        echo "=== $V"
        "$PY" model_design/train_ablation.py --variant "$V" --workers 20 --preload-workers 20 \
            || echo "FAILED $V"
    done
    echo "A3B A4B DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -w "ended(29157969)" -J argos_a3b4b "$SELF --node $*"
