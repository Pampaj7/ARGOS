#!/bin/bash
# The matched-policy closure, out of domain.
#
# The mechanism argument -- that the gain is not temporal smoothing -- rests on a table
# that exists only on SCARED-C D2. A reviewer asked for it on held-out data; our own
# freeze forbids that, because experimental_closure.py permits one decision-pinned method
# on D7 and running fifteen policies there is exactly what the freeze exists to prevent.
#
# DRENDS is the arena the protocol does allow ("external analysis only; no tuning"), and
# it is also the only arena that is genuinely out of domain -- D2 and D7 are both SCARED-C,
# same tissue, same rig, so a reviewer reads them as in-domain and is right to.
#
# Five representative policies on RAFT-Stereo across all five recordings. None loads a
# checkpoint. TETHER's own DRENDS numbers already exist in results/drends_a2/seed0.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/drends_closure
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
    for P in fixed_w0_5_h4 ema3_h4 fb_confidence_h4 warped_recurrent_h4 warped_raw_previous_h1; do
        DEST="$OUT/$P"
        [ -f "$DEST/summary.json" ] && { echo "=== $P already done"; continue; }
        echo "=== $P"
        "$PY" -m model_design.comparison.drends_backbone_transfer \
            --module "model_design.comparison.experimental_policies:factory_$P" \
            --backbone RAFT-Stereo --device cuda:0 --output "$DEST" || echo "FAILED $P"
    done
    echo "DRENDS CLOSURE DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_dclosure "$SELF --node $*"
