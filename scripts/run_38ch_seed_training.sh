#!/bin/bash
# Trains A3b and A6 at one seed (argument), sequentially, for the two-base finding.
set -u
SEED=${1:?usage: run_38ch_seed_training.sh <seed>}
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${2:-}" = "--node" ]; then
    BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v busy="$BUSY" '{used=index(busy,$2)>0; print used, -$3, $1}' \
        | sort -k1,1n -k2,2n | head -1 | awk '{print $3}')
    echo "picked physical GPU $CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    for V in A3b_single_resolution_38ch A6_geometry_only; do
        RUN="$ROOT/model_design/training_runs/ablation_${V}_seed_${SEED}"
        EPOCHS_DONE=$(( $(wc -l < "$RUN/training_history.csv" 2>/dev/null || echo 1) - 1 ))
        if [ "$EPOCHS_DONE" -ge 12 ]; then echo "=== $V seed $SEED already trained"; continue; fi
        echo "=== $V seed $SEED"
        "$PY" model_design/train_ablation_seed.py --variant "$V" --seed "$SEED" \
            --workers 20 --preload-workers 20 || echo "FAILED $V seed $SEED"
    done
    echo "SEED $SEED TRAINING DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J "argos_s${SEED}train" "$SELF $SEED --node"
