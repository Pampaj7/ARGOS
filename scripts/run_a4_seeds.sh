#!/bin/bash
# A4 at seeds 1 and 2.
#
# A4 relaxes the convex-fusion constraint -- the one inductive bias the paper keeps from
# CODD -- and it is the only ablation still standing on a single run. Our own declared
# threshold is 0.37 points over three seeds, and A4's held-out Bad1 margin is 0.87, so the
# number that decides whether the constraint is load-bearing is currently unreadable
# against the rule we wrote for reading it.
#
# Seeds 1 and 2 are the only pre-registered deviation; train_ablation_seed.py enforces that.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
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
    for S in ${SEEDS:-1 2}; do
        RUN="$ROOT/model_design/training_runs/ablation_A4_relaxed_convexity_seed_$S"
        EPOCHS_DONE=$(( $(wc -l < "$RUN/training_history.csv" 2>/dev/null || echo 1) - 1 ))
        if [ "$EPOCHS_DONE" -ge 12 ]; then echo "=== seed $S already trained ($EPOCHS_DONE epochs)"; continue; fi
        [ "$EPOCHS_DONE" -gt 0 ] && echo "=== seed $S resuming from epoch $EPOCHS_DONE"
        echo "=== A4 seed $S"
        "$PY" model_design/train_ablation_seed.py --variant A4_relaxed_convexity --seed "$S" \
            --workers 20 --preload-workers 20 || echo "FAILED seed $S"
    done
    echo "A4 SEEDS DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -env "all, SEEDS=${SEEDS:-1 2}" -J argos_a4seed "$SELF --node $*"
