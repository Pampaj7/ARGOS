#!/bin/bash
# Evaluate the per-backbone specialists on their own backbone's held-out cells.
#
# The control's endpoint, per specialist_control_declaration.json: D7 relative EPE and
# Bad1 reduction for the arm's own backbone, against the shipped generalist on the same
# cells, under the definitive protocol with no new support and no new aggregation. D2 is
# evaluated too because that is the split the specialist selected its checkpoint on, and
# reporting only the arena where it was chosen would be the selection optimism this paper
# spends a section on.
#
# Only arms whose training has finished are evaluated: a 36-epoch arm read at epoch 20
# would score a checkpoint the declaration does not describe. Re-runnable -- finished
# arms are skipped by the presence of their definitive_table.csv.
#
# --scared-backbones is not optional here. Its default is all five, so omitting it scores
# each specialist on every backbone -- thirty-five cells per arm instead of seven, and an
# off-diagonal experiment nobody declared, chosen by a forgotten flag rather than on
# purpose. The declared endpoint is the diagonal: each arm on the backbone it trained for.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/specialist_eval
EPOCHS=36
SELF=$(readlink -f "$0")
declare -A FACTORY=(
    [RAFT-Stereo]=factory_raft
    [StereoAnywhere]=factory_stereoanywhere
    [S2M2-S]=factory_s2m2
    [CREStereo]=factory_crestereo
    [Fast-FoundationStereo]=factory_fastfoundation
)
if [ "${1:-}" = "--node" ]; then
    shift
    export CUDA_VISIBLE_DEVICES=${PIN_GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
    cd "$ROOT" || exit 1
    for B in "${!FACTORY[@]}"; do
        RUN="$ROOT/model_design/training_runs/specialist_$B"
        DONE=$(cat "$RUN/training_history.csv" 2>/dev/null | wc -l)
        DONE=$(( DONE > 0 ? DONE - 1 : 0 ))
        if [ "$DONE" -lt "$EPOCHS" ]; then
            echo "=== $B still training ($DONE/$EPOCHS), not evaluated"; continue
        fi
        if [ -f "$OUT/$B/definitive_table.csv" ]; then echo "=== $B already evaluated"; continue; fi
        [ -d "$OUT/$B" ] && mv "$OUT/$B" "$OUT/$B.incomplete-$(date +%s)"
        echo "=== $B (${FACTORY[$B]})"
        "$PY" model_design/comparison/run_definitive_evaluation.py --datasets scared-d2 scared-d7 \
            --module "model_design.comparison.specialist_h4:${FACTORY[$B]}" --device cuda:0 \
            --scared-backbones "$B" \
            --output "$OUT/$B" || echo "FAILED $B"
    done
    echo "SPECIALIST EVAL DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=1:mode=shared" -J argos_speceval "$SELF --node $*"
