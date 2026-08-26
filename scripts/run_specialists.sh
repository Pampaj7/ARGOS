#!/bin/bash
# Per-backbone specialist control: the shipped 38-channel head trained on ONE backbone.
#
# The control contribution (i) needs. If a module trained for a single estimator does not
# beat the one shared checkpoint on that estimator, generality is free; if it does, the
# paper owes the reader the size of the toll. Pre-registered in
# model_design/specialist_control_declaration.json before any arm was launched, including
# the declared asymmetry: every deviation favours the specialist (matched gradient budget
# at 36 epochs, three times the selection points, validation on its own backbone).
#
# Two lanes, one GPU each, launched as two jobs with PIN_GPU so they cannot race for the
# same device the way three same-snapshot pickers did on 2026-08-21:
#
#   PIN_GPU=0 nohup scripts/run_specialists.sh a > logs/spec_a.log 2>&1 &
#   PIN_GPU=1 nohup scripts/run_specialists.sh b > logs/spec_b.log 2>&1 &
#
# Lane a carries three arms (~8.5 h), lane b two (~5.7 h); the seen backbone the paper
# quotes most and the strongest unseen cell run first in their lanes, so an interrupted
# night still lands the two arms worth reading. Each arm resumes from its own history, so
# a relaunch after the 12 h wall continues rather than restarts. The pend is interactive
# (p1i requires bsub -I), so launch under nohup or the pend dies with the shell.
set -u
LANE=${1:?usage: [PIN_GPU=n] run_specialists.sh <a|b>}
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
PIN=${PIN_GPU:-0}
case "$LANE" in
    a) ARMS="RAFT-Stereo StereoAnywhere S2M2-S" ;;
    b) ARMS="Fast-FoundationStereo CREStereo" ;;
    *) echo "lane must be a or b"; exit 1 ;;
esac
EPOCHS=36
if [ "${2:-}" = "--node" ]; then
    if [ -n "$PIN" ]; then CUDA_VISIBLE_DEVICES=$PIN; else
    BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v busy="$BUSY" '{used=index(busy,$2)>0; print used, -$3, $1}' \
        | sort -k1,1n -k2,2n | head -1 | awk '{print $3}')
    fi
    echo "lane $LANE on physical GPU $CUDA_VISIBLE_DEVICES: $ARMS"
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    for B in $ARMS; do
        RUN="$ROOT/model_design/training_runs/specialist_$B"
        DONE=$(( $(wc -l < "$RUN/training_history.csv" 2>/dev/null || echo 1) - 1 ))
        if [ "$DONE" -ge "$EPOCHS" ]; then echo "=== $B already trained ($DONE epochs)"; continue; fi
        [ "$DONE" -gt 0 ] && echo "=== $B resuming from epoch $DONE"
        echo "=== $B"
        "$PY" model_design/train_specialist.py --backbone "$B" \
            --workers 20 --preload-workers 20 || echo "FAILED $B"
    done
    echo "SPECIALIST LANE $LANE DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
# One GPU, not two: p1i caps a user at 4 physical GPUs, and a lane that reserves two
# while pinning one spends a slot that another job could be running in. Observed on
# 2026-08-26, when the reservation was what kept the DRENDS seed fills pending.
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=1:mode=shared" -J "argos_spec_$LANE" "$SELF $LANE --node"
