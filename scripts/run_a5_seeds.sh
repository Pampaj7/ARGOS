#!/bin/bash
# A5 (no FB cue in training) at seeds 1 and 2, then their evaluations: the last
# single-seed arm of the train-with/deploy-without story.
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
    for S in 1 2; do
        RUN="$ROOT/model_design/training_runs/ablation_A5_no_fb_cue_seed_${S}"
        EPOCHS_DONE=$(( $(wc -l < "$RUN/training_history.csv" 2>/dev/null || echo 1) - 1 ))
        if [ "$EPOCHS_DONE" -ge 12 ]; then echo "=== A5 seed $S already trained"; continue; fi
        echo "=== A5 seed $S train"
        "$PY" model_design/train_ablation_seed.py --variant A5_no_fb_cue --seed "$S" \
            --workers 20 --preload-workers 20 || echo "FAILED A5 seed $S train"
    done
    for S in 1 2; do
        DEST=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/ablation_eval/a5_seed$S
        [ -d "$DEST/runs/scared-d7" ] && { echo "=== a5_seed$S already evaluated"; continue; }
        echo "=== a5_seed$S eval"
        "$PY" model_design/comparison/run_definitive_evaluation.py --datasets scared-d2 scared-d7 \
            --module "model_design.comparison.ablation_h4:factory_a5_seed$S" --device cuda:0 \
            --output "$DEST" || echo "FAILED a5_seed$S eval"
    done
    echo "A5 SEEDS DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_a5seeds "$SELF --node $*"
