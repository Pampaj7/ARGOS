#!/bin/bash
# A1 (no appearance, 78ch, 142-channel base) at seeds 1 and 2: the ladder's middle rung
# joins the seed treatment now that the two-base finding makes the ladder load-bearing.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
PIN=${PIN_GPU:-}
if [ "${1:-}" = "--node" ]; then
    if [ -n "$PIN" ]; then CUDA_VISIBLE_DEVICES=$PIN; else
    BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v busy="$BUSY" '{used=index(busy,$2)>0; print used, -$3, $1}' \
        | sort -k1,1n -k2,2n | head -1 | awk '{print $3}')
    fi
    echo "picked physical GPU $CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    for S in 1 2; do
        RUN="$ROOT/model_design/training_runs/ablation_A1_no_appearance_seed_${S}"
        EPOCHS_DONE=$(( $(wc -l < "$RUN/training_history.csv" 2>/dev/null || echo 1) - 1 ))
        if [ "$EPOCHS_DONE" -ge 12 ]; then echo "=== A1 seed $S already trained"; continue; fi
        echo "=== A1 seed $S"
        "$PY" model_design/train_ablation_seed.py --variant A1_no_appearance --seed "$S" \
            --workers 20 --preload-workers 20 || echo "FAILED A1 seed $S"
    done
    echo "A1 SEEDS TRAINING DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_a1seeds "$SELF --node $*"
