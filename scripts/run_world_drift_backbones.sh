#!/bin/bash
# Does world-frame drift improve where the raw prediction is poor, or only on D7?
#
# On RAFT-Stereo the split is clean: D2, where raw excess drift is 0.18-0.22 mm, gets
# worse (-6.8 to -22.0%); D7, where it is 0.55-1.11 mm, gets better (+8.7 to +15.4%).
# Two readings fit that. Either the module helps where the raw signal is poor, which is a
# statement about raw quality, or it helps on D7, which is a statement about a dataset and
# would not generalise at all.
#
# Backbones separate them. Their raw quality differs on the *same* sequences, so if a
# weaker backbone turns D2 positive the condition is raw quality; if D2 stays negative for
# every backbone, it is the dataset and the paper must say so.
#
# TETHER only: the point is a property of the shipped model, and running the ablation head
# here would double the cost to answer a question nobody asked about it.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/world_frame_drift
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    # RAFT-Stereo is already done for every sequence and is not repeated.
    for BB in ${BACKBONES:-S2M2-S StereoAnywhere CREStereo Fast-FoundationStereo}; do
        for SEQ in ${SEQS:-dataset_2_keyframe_2 dataset_2_keyframe_3 dataset_2_keyframe_4 dataset_7_keyframe_1}; do
            DEST="$OUT/${SEQ}_a2_${BB}.json"
            if [ -f "$DEST" ]; then echo "=== $SEQ / $BB already done"; continue; fi
            echo "=== $SEQ / $BB"
            "$PY" scripts/world_frame_drift.py --sequence "$SEQ" --backbone "$BB" \
                --frames 200 --min-views 3 --device cuda:0 \
                --module model_design.comparison.ablation_h4:factory_a2 \
                --output "$DEST" || echo "FAILED $SEQ $BB"
        done
    done
    echo "WORLD DRIFT BACKBONES DONE"
    exit 0
fi
# -env rejects an empty assignment, so the defaults are resolved here rather than left to
# the node half to substitute.
export ESUB_BYPASS=1 ESUB_QUIET=1
BACKBONES=${BACKBONES:-S2M2-S StereoAnywhere CREStereo Fast-FoundationStereo}
SEQS=${SEQS:-dataset_2_keyframe_2 dataset_2_keyframe_3 dataset_2_keyframe_4 dataset_7_keyframe_1}
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=10GB]" \
     -gpu "num=1:mode=shared" -env "all, BACKBONES=$BACKBONES, SEQS=$SEQS" \
     -J argos_drift_bb "$SELF --node $*"
