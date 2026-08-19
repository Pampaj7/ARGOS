#!/bin/bash
# World-frame reconstruction drift on SCARED-C, raw against refined, for both heads.
#
# The downstream evidence the review says the paper needs for a robotics venue. It measures
# what a map accumulates -- how much frames disagree about the same world point -- rather
# than what a frame scores, and it is possible at all because SCARED-C ships corrected
# per-frame poses, which the paper spent a day claiming it does not.
#
# Ground truth is the floor and must come out near zero: it is one keyframe's geometry
# propagated by exactly these poses, so undoing the propagation has to collapse it. If a run
# reports a large ground-truth spread, the result is wrong, not interesting.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/world_frame_drift
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
    echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
    # D2 is development and D7 is held out. The measurement is the same either way, and a
    # negative result that only exists on the split we tuned on would be the weakest form of
    # it -- so both are run and reported together.
    SEQS="dataset_2_keyframe_2 dataset_2_keyframe_3 dataset_2_keyframe_4"
    SEQS="$SEQS dataset_7_keyframe_1 dataset_7_keyframe_2 dataset_7_keyframe_3 dataset_7_keyframe_4"
    for SEQ in $SEQS; do
        for SPEC in "canonical:model_design.comparison.canonical_h4_masked:factory" \
                    "a2:model_design.comparison.ablation_h4:factory_a2"; do
            NAME="${SPEC%%:*}"; MODULE="${SPEC#*:}"
            DEST="$OUT/${SEQ}_${NAME}.json"
            if [ -f "$DEST" ]; then echo "=== $SEQ / $NAME already done"; continue; fi
            echo "=== $SEQ / $NAME"
            "$PY" scripts/world_frame_drift.py --sequence "$SEQ" --backbone RAFT-Stereo \
                --frames 200 --min-views 3 --module "$MODULE" --device cuda:0 \
                --output "$DEST" || echo "FAILED $SEQ $NAME"
        done
    done
    echo "WORLD DRIFT DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=10GB]" \
     -gpu "num=1:mode=shared" -J argos_worlddrift "$SELF --node $*"
