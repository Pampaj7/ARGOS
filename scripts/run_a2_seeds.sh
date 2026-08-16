#!/bin/bash
# A2 with seeds 1 and 2, so it gets the same three-seed treatment as the canonical
# model. The decision split is D2; D7 is not consulted for this comparison.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    cd "$ROOT" || exit 1
    for S in 1 2; do
        G=$((S-1))
        CUDA_VISIBLE_DEVICES=$G nohup "$PY" model_design/train_ablation_seed.py \
            --variant A2_no_learned_evidence --seed "$S" \
            > /dtu/p1/leopam/ARGOS/logs/train_A2_seed$S.log 2>&1 &
        echo "A2 seed $S -> GPU $G (pid $!)"
        sleep 5
    done
    wait
    echo "A2 SEEDS DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=2:mode=shared" -J argos_a2seeds "$SELF --node $*"
