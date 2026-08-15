#!/bin/bash
# Canonical H4 seeds 1 and 2, one per H100, in a single p1i job.
# Pre-registered in model_design/multiseed_preregister.json.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    cd "$ROOT" || exit 1
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
    for S in 1 2; do
        G=$((S-1))
        CUDA_VISIBLE_DEVICES=$G nohup "$PY" model_design/train_seed.py --seed "$S" \
            > /dtu/p1/leopam/ARGOS/logs/train_seed_$S.log 2>&1 &
        echo "seed $S -> physical GPU $G (pid $!)"
        sleep 5
    done
    wait
    echo "SEEDS DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=2:mode=shared" -J argos_seeds "$SELF --node $*"
