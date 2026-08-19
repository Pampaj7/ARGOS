#!/bin/bash
# A3 at seeds 1 and 2, the training the review asks for alongside A2's.
#
# A2 got its three seeds and moved from "one run, suggestive" to a promotion. A3 sits at the
# same distance from the canonical mean on a single run and has never had the same treatment,
# so the paper still describes it with a caveat that A2 no longer needs. Two training runs
# settle whether "none of the four design decisions is load-bearing" is a claim about the
# architecture or about single-seed noise.
#
# Same locked recipe, seed the only deviation, as pre-registered in multiseed_preregister.json.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    # num=2 below is what lets this pick: with num=1 LSF grants one device and naming the
    # other leaves CUDA with nothing.
    # Prefer a GPU with no compute process at all, and fall back to most-free-memory. Free
    # memory alone is a poor criterion when jobs dispatch minutes apart: they read the same
    # snapshot and pile onto the same device while the other sits at 0% -- observed twice.
    BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v busy="$BUSY" '{used=index(busy,$2)>0; print used, -$3, $1}' \
        | sort -k1,1n -k2,2n | head -1 | awk '{print $3}')
    echo "picked physical GPU $CUDA_VISIBLE_DEVICES of: $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr '\n' ' ')"
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    for S in ${SEEDS:-1 2}; do
        RUN="$ROOT/model_design/training_runs/ablation_A3_single_resolution_seed_$S"
        # Completion is twelve epochs of history, not the mere existence of a checkpoint.
        # best_validation.pt appears after the first epoch, so the old test declared a run
        # finished the moment it started and made an interrupted seed unresumable.
        EPOCHS_DONE=$(( $(wc -l < "$RUN/training_history.csv" 2>/dev/null || echo 1) - 1 ))
        if [ "$EPOCHS_DONE" -ge 12 ]; then echo "=== seed $S already trained ($EPOCHS_DONE epochs)"; continue; fi
        [ "$EPOCHS_DONE" -gt 0 ] && echo "=== seed $S resuming from epoch $EPOCHS_DONE"
        echo "=== A3 seed $S"
        "$PY" model_design/train_ablation_seed.py --variant A3_single_resolution --seed "$S" \
            --workers 12 --preload-workers 12 || echo "FAILED seed $S"
    done
    echo "A3 SEEDS DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -env "all, SEEDS=${SEEDS:-1 2}" -J argos_a3seed "$SELF --node $*"
