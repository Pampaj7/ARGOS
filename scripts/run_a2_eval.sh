#!/bin/bash
# A3 (single-resolution context branch), pre-registered. Trains, then evaluates
# on SCARED-C D2 and D7 under the unmodified definitive protocol.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    "$PY" model_design/comparison/run_definitive_evaluation.py --datasets scared-d2 scared-d7 \
        --module model_design.comparison.ablation_h4:factory_a2 --device cuda:0 \
        --output /dtu/p1/leopam/ARGOS/ARGOS_hand/results/ablation_eval/a2 || echo "EVAL FAILED"
    echo "A3 DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=2:mode=shared" -J argos_a2eval "$SELF --node $*"
