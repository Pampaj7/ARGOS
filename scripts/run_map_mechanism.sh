#!/bin/bash
# Two questions about the world-frame result, both answerable from predictions we can
# already produce, neither needing a new model.
#
# (1) MECHANISM. The paper says the mapping benefit is conditional and admits it cannot
#     show why. The candidate explanation is that a map does not pay the per-frame mean
#     error, it pays the view-dependent component: raw stereo error is partly systematic
#     and cancels under multi-view fusion, while the alignment error the warp introduces
#     is frame-pair dependent and does not. The blend trades one for the other, so it wins
#     where raw excess is large (D7, 0.518-1.113 mm) and loses where raw excess is the
#     same order as the alignment residual (D2, 0.171-0.290).
#
#     Two probes separate the warp from the recurrence:
#       warped_raw_previous_h1 -- aligned raw memory, no recurrence: the cost of one warp.
#       the horizon sweep H=1,2,6 on the shipped head: if the D2 degradation grows with H
#       it is alignment error accumulated by recurrence; if it is already full at H=1 it
#       is the single warp.
#
# (2) DEGENERATE CONTROL. View disagreement is a consistency measure, not an accuracy one,
#     and our own contribution (iii) says a temporally correlated prediction can reduce a
#     consistency score while staying wrong. We ran that control on DTCE and on the
#     no-reference score and never on this metric. copy_previous and warped_previous are
#     the two degenerate policies; if either wins the map metric, the D7 result has to be
#     restated.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/world_frame_drift
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
    run () {  # tag  module
        for SEQ in dataset_2_keyframe_2 dataset_2_keyframe_3 dataset_2_keyframe_4 \
                   dataset_7_keyframe_1 dataset_7_keyframe_2 dataset_7_keyframe_3 dataset_7_keyframe_4; do
            DEST="$OUT/${SEQ}_$1.json"
            [ -f "$DEST" ] && { echo "  $1 $SEQ already done"; continue; }
            echo "=== $1 $SEQ"
            "$PY" scripts/world_frame_drift.py --sequence "$SEQ" --backbone RAFT-Stereo \
                --frames 200 --module "$2" --device cuda:0 --output "$DEST" || echo "FAILED $1 $SEQ"
        done
    }
    run copyprev   model_design.comparison.degenerate_policies:factory
    run warpedprev model_design.comparison.degenerate_policies:factory_warped
    run a2h1       model_design.comparison.ablation_horizons:factory_a2_h1
    run a2h2       model_design.comparison.ablation_horizons:factory_a2_h2
    run a2h6       model_design.comparison.ablation_horizons:factory_a2_h6
    echo "MAP MECHANISM DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_mapmech "$SELF --node $*"
