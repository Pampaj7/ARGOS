#!/bin/bash
# A2 on DRENDS, three seeds x three backbones. The promotion decision was already
# made on D2 under the pre-registration; this is reporting for the promoted model,
# not a selection criterion, and it is reported whichever way it comes out.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/drends_a2
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
    cd "$ROOT" || exit 1
    for SEEDSPEC in "factory_a2:seed0" "factory_a2_seed1:seed1" "factory_a2_seed2:seed2"; do
        F="${SEEDSPEC%%:*}"; S="${SEEDSPEC##*:}"
        for B in RAFT-Stereo CREStereo Fast-FoundationStereo; do
            echo "=== $S / $B"
            "$PY" -m model_design.comparison.drends_backbone_transfer --device cuda:0 \
                --backbone "$B" --module "model_design.comparison.ablation_h4:$F" \
                --recordings Vid10_Liver_Med Vid11_Liver_High Vid12_Pancreas_Ext Vid13_Pancreas_Med Vid14_Pancreas_High \
                --output "$OUT/$S/$B" || echo "FAILED $S $B"
        done
    done
    echo "A2 DRENDS DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=2:mode=shared" -J argos_a2drends "$SELF --node $*"
