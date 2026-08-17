#!/bin/bash
# Run BiDAStabilizer over the five exported DRENDS boundaries.
#
# Second half of the out-of-domain head-to-head. The published comparison ran on SCARED-C
# D2 -- the split our checkpoint was selected on, while BiDAStabilizer had never seen the
# domain -- and the paper now declares that and reads the margin as an upper bound on our
# advantage. On DRENDS neither method has seen the domain.
#
# The stabiliser reads RGB only and computes its own RAFT-Stereo robust disparity, which
# it writes to raw.npz. That array, not our cached middlebury one, is the shared input:
# TETHER is scored on it afterwards. Running BiDA first is therefore not an ordering
# preference, it is the only order that produces a common boundary.
set -u
ROOT=/dtu/p1/leopam/ARGOS
EXT=$ROOT/ARGOS_hand/external_comparison
BOUNDARY=$EXT/results/drends_boundary
OUT=$EXT/results/bidastabilizer_drends
PY=$ROOT/.miniconda/envs/argos/bin/python
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    # run_external_evaluation.py refuses physical GPU 0 -- it is reserved, and the D2
    # runner hardcodes device 1 for the same reason. Picking the freest device, as the
    # other launchers here do, selected GPU 0 and every recording failed at once. The
    # job therefore requests two GPUs so that device 1 exists at all.
    if ! nvidia-smi --query-gpu=index --format=csv,noheader | grep -qx 1; then
        echo "device 1 not visible; external comparisons may not run on GPU 0" >&2; exit 1
    fi
    export ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$EXT" || exit 1
    for REC in Vid10_Liver_Med Vid11_Liver_High Vid12_Pancreas_Ext Vid13_Pancreas_Med Vid14_Pancreas_High; do
        DEST="$OUT/$REC"
        # Resumable: a recording whose refined output already exists is left alone, so a
        # requeue after a walltime kill does not redo finished work or overwrite it.
        if [ -f "$DEST/refined.npz" ]; then echo "=== $REC already complete"; continue; fi
        echo "=== $REC"
        mkdir -p "$DEST"
        "$PY" run_external_evaluation.py --method bidastabilizer_bidirectional_offline \
            --input "$BOUNDARY/$REC/seed.npz" \
            --raw-output "$DEST/raw.npz" --output "$DEST/refined.npz" \
            --python "$PY" --purpose DRENDS_FULL_DIAGNOSTIC || echo "FAILED $REC"
    done
    echo "BIDA DRENDS DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=2:mode=shared" -J argos_bidadrends "$SELF --node $*"
