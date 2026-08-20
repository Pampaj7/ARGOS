#!/bin/bash
# The support-contract ablation on the head the paper ships, paired.
#
# The paper reports the confinement result -- unmasked depth RMSE regressing by +110.4%,
# confined improving by -3.2% -- from the 142-channel configuration, on 14 and 15 cells,
# and says the arms are not paired. Both halves of that caveat are fixable: the shipped
# head has an unconfined twin now, and running it over the same 15 DRENDS cells as
# results/drends_a2/seed0 makes the comparison paired on one checkpoint.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/drends_a2_unconfined
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
    for BB in RAFT-Stereo S2M2-S StereoAnywhere CREStereo Fast-FoundationStereo; do
        echo "=== $BB"
        "$PY" -m model_design.comparison.drends_backbone_transfer \
            --module model_design.comparison.ablation_h4:factory_a2_unconfined \
            --backbone "$BB" --device cuda:0 --output "$OUT/$BB" || echo "FAILED $BB"
    done
    echo "MASKING A2 DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_maska2 "$SELF --node $*"
