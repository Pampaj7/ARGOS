#!/bin/bash
# Pose-perturbation sensitivity for the accumulated-map result.
#
# The D2/D7 sign reversal is the paper's most-contested finding, and the objection worth
# answering is not that its explanation is unproven -- the paper already says that -- but
# that nobody has shown the effect clears the pose-noise floor. The held-out gain is 11.7%
# of an excess of 0.5-1.1 mm, so of order 0.1 mm, and the poses carry a residual the paper
# never quantifies.
#
# Two things come out of one sweep. The ground-truth cloud is one keyframe's structured
# light moved by these very poses, so its world-frame spread is already an empirical
# measurement of that residual; and perturbing the poses by a known sigma says at what
# magnitude the sign of the reduction stops surviving.
#
# The seven RAFT-Stereo cells the horizon sweep reports, at the same 200 frames, so the
# zero-noise column of every run must reproduce results/world_frame_drift/*_a2.json exactly.
# That reproduction is the test that the refactor did not change what is measured; it is
# checked by scripts/check_pose_noise_baseline.py after the run.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/world_frame_pose_noise
SEQS="dataset_2_keyframe_2 dataset_2_keyframe_3 dataset_2_keyframe_4 dataset_7_keyframe_1 dataset_7_keyframe_2 dataset_7_keyframe_3 dataset_7_keyframe_4"
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    export CUDA_VISIBLE_DEVICES=${PIN_GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
    cd "$ROOT" || exit 1
    for S in $SEQS; do
        if [ -f "$OUT/$S.json" ]; then echo "=== $S already done"; continue; fi
        echo "=== $S"
        "$PY" scripts/world_frame_drift.py --sequence "$S" --backbone RAFT-Stereo \
            --frames 200 --device cuda:0 --pose-noise --noise-repeats 5 \
            --output "$OUT/$S.json" || echo "FAILED $S"
    done
    echo "POSE NOISE DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=1:mode=shared" -J argos_posenoise "$SELF --node $*"
