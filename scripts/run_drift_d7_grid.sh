#!/bin/bash
# Complete the held-out half of the world-frame drift grid.
#
# The paper reports "23 cells in which the sign never changes: all 15 D2 cells negative and
# all 8 D7 cells positive". The asymmetry is not a design: D2 got all five backbones on all
# three sequences, D7 got all five on keyframe 1 and RAFT-Stereo alone on keyframes 2-4. So
# the favourable half of a sign-consistency claim rests on a smaller and differently-shaped
# support than the unfavourable half, and the paper never says so. Twelve cells fix it.
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
    for SEQ in dataset_7_keyframe_2 dataset_7_keyframe_3 dataset_7_keyframe_4; do
        for BB in S2M2-S StereoAnywhere CREStereo Fast-FoundationStereo; do
            DEST="$OUT/${SEQ}_a2_${BB}.json"
            [ -f "$DEST" ] && { echo "=== $SEQ $BB already done"; continue; }
            echo "=== $SEQ $BB"
            "$PY" scripts/world_frame_drift.py --sequence "$SEQ" --backbone "$BB" \
                --frames 200 --module model_design.comparison.ablation_h4:factory_a2 \
                --device cuda:0 --output "$DEST" || echo "FAILED $SEQ $BB"
        done
    done
    echo "DRIFT D7 GRID DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_driftd7 "$SELF --node $*"
