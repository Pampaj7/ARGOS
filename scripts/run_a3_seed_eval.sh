#!/bin/bash
# Evaluate A3 seeds 1 and 2, so the one deviation that beat its base on a single run gets
# the same three-seed treatment everything else in the ablation table has.
#
# A3 runs the context branch at full resolution instead of a quarter. On one run it scored
# 5.61/5.85 against the 5.06/5.15 of the configuration it deviates from -- 0.70 points on
# the pre-registered endpoint, nearly twice the 0.37-point seed spread that was declared as
# the threshold. Either that survives two more seeds, in which case the paper concedes that
# a design choice it did not adopt is better on the endpoint it pre-registered, or it does
# not, in which case the single run was seed luck. Both are reportable; not knowing is not.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
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
    for S in 1 2; do
        DEST=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/ablation_eval/a3_seed$S
        if [ -f "$DEST/runs/scared-d7/model_design_comparison_ablation_h4__factory_a3_seed$S/run_manifest.json" ]; then
            echo "=== A3 seed $S already evaluated"; continue; fi
        echo "=== A3 seed $S"
        "$PY" model_design/comparison/run_definitive_evaluation.py --datasets scared-d2 scared-d7 \
            --module "model_design.comparison.ablation_h4:factory_a3_seed$S" --device cuda:0 \
            --output "$DEST" || echo "FAILED seed $S"
    done
    echo "A3 SEED EVAL DONE"
    exit 0
fi
# rusage[mem] is per core, so -n 4 with 40GB reserves 160GB and starves the node against
# itself. 12GB per core is what these evaluations actually use.
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_a3seedeval "$SELF --node $*"
