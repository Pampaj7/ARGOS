#!/bin/bash
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
PIN=${PIN_GPU:-}
if [ "${1:-}" = "--node" ]; then
    if [ -n "$PIN" ]; then CUDA_VISIBLE_DEVICES=$PIN; else CUDA_VISIBLE_DEVICES=1; fi
    echo "picked physical GPU $CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    for F in a3b_seed1 a3b_seed2 a6_seed1 a1_seed1; do
        DEST=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/ablation_eval/$F
        [ -d "$DEST/runs/scared-d7" ] && { echo "=== $F already done"; continue; }
        echo "=== $F"
        "$PY" model_design/comparison/run_definitive_evaluation.py --datasets scared-d2 scared-d7 \
            --module "model_design.comparison.ablation_h4:factory_$F" --device cuda:0 \
            --output "$DEST" || echo "FAILED $F"
    done
    echo "READY SEED EVALS DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_rdyeval "$SELF --node $*"
