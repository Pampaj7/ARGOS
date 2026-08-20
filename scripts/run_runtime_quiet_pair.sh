#!/bin/bash
# Paired quiet re-measurement of shipped vs full-res context (T2.2 retraction follow-up).
# Must run with no other GPU work on the node: the first A3 timing was contaminated by
# contention (identical stages inflated 4x while the changed stage moved 0.3 ms).
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    for M in factory_a2 factory_a3; do
        "$PY" scripts/measure_runtime.py --device cuda:0 \
            --module "model_design.comparison.ablation_h4:$M" \
            --output "../results/runtime/quiet_pair_$M.json" || echo "FAILED $M"
    done
    echo "QUIET PAIR DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_rtquiet "$SELF --node $*"
