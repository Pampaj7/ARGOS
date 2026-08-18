#!/bin/bash
# The development closure re-run for the promoted A2 head, seeds 0-2.
#
# Only the six learned-head rows are re-run. The fifteen baseline policies blend raw
# against flow-aligned raw with a fixed, EMA or confidence weight and load no checkpoint,
# so they are identical whichever head the paper ships; the raw-vs-memory oracle is GT-only
# for the same reason. Re-canonicalising therefore costs six method-configs per seed, not
# twenty-one, which is why this is a night rather than a week.
#
# Each seed writes its own root. A head that landed on results/d2 would overwrite the
# canonical closure the paper still cites while the promotion is being verified.
#
# The runner is promoted_head_closure.py, not experimental_closure.py: the latter and
# canonical_horizons.py are both pinned by the freeze manifest, and editing either made the
# protocol refuse to run. The frozen experiment keeps its bytes.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
HEAD=A2_no_learned_evidence
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    for SEED in 0 1 2; do
        SUF="${HEAD}"; [ "$SEED" != "0" ] && SUF="${HEAD}_seed${SEED}"
        OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/definitive_evaluation/experimental_closure/d2_${SUF}
        if [ -f "$OUT/summary.csv" ]; then echo "=== seed $SEED already complete"; continue; fi
        echo "=== closure seed $SEED"
        SEED_ARG=(--head-seed "$SEED")
        [ "$SEED" = "0" ] && SEED_ARG=()   # seed 0 trained without a seed suffix in its run dir
        "$PY" -m model_design.comparison.promoted_head_closure \
            --head "$HEAD" "${SEED_ARG[@]}" --device cuda:0 || echo "FAILED seed $SEED"
    done
    echo "CLOSURE A2 DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=1:mode=shared" -J argos_closurea2 "$SELF --node $*"
