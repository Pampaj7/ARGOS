#!/bin/bash
# Specialists out of domain, on their own backbone.
#
# Added by the 2026-08-27 addendum to specialist_control_declaration.json, before any
# specialist result existed. D7 stays the registered endpoint; DRENDS is reported
# alongside it because D7 is held-out sequences of the dataset the specialist trained on
# -- the arena where a single-backbone module is strongest -- and reporting the control
# only there would be arena selection.
#
# Two lanes so the four finished arms cover in half the wall time:
#   PIN_GPU=0 nohup scripts/run_specialist_drends.sh a > logs/specdrends_a.log 2>&1 &
#   PIN_GPU=0 nohup scripts/run_specialist_drends.sh b > logs/specdrends_b.log 2>&1 &
#
# An arm short of its 36 epochs is skipped, not evaluated at whatever checkpoint exists:
# scoring a half-trained model as the declared one is the failure this guard exists for.
set -u
LANE=${1:?usage: [PIN_GPU=n] run_specialist_drends.sh <a|b>}
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/specialist_drends
RECS="Vid10_Liver_Med Vid11_Liver_High Vid12_Pancreas_Ext Vid13_Pancreas_Med Vid14_Pancreas_High"
EPOCHS=36
SELF=$(readlink -f "$0")
case "$LANE" in
    a) ARMS="RAFT-Stereo StereoAnywhere S2M2-S" ;;
    b) ARMS="Fast-FoundationStereo CREStereo" ;;
    *) echo "lane must be a or b"; exit 1 ;;
esac
factory_of() {
    case "$1" in
        RAFT-Stereo) echo factory_raft ;;
        StereoAnywhere) echo factory_stereoanywhere ;;
        S2M2-S) echo factory_s2m2 ;;
        CREStereo) echo factory_crestereo ;;
        Fast-FoundationStereo) echo factory_fastfoundation ;;
    esac
}
if [ "${2:-}" = "--node" ]; then
    export CUDA_VISIBLE_DEVICES=${PIN_GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "lane $LANE on $(hostname) gpu=$CUDA_VISIBLE_DEVICES: $ARMS"
    cd "$ROOT" || exit 1
    for B in $ARMS; do
        RUN="$ROOT/model_design/training_runs/specialist_$B"
        DONE=$(cat "$RUN/training_history.csv" 2>/dev/null | wc -l)
        DONE=$(( DONE > 0 ? DONE - 1 : 0 ))
        if [ "$DONE" -lt "$EPOCHS" ]; then echo "=== $B still training ($DONE/$EPOCHS), skipped"; continue; fi
        if [ -f "$OUT/$B/summary.json" ]; then echo "=== $B already evaluated"; continue; fi
        [ -d "$OUT/$B" ] && mv "$OUT/$B" "$OUT/$B.incomplete-$(date +%s)"
        echo "=== $B ($(factory_of $B))"
        "$PY" -m model_design.comparison.drends_backbone_transfer --device cuda:0 \
            --backbone "$B" --module "model_design.comparison.specialist_h4:$(factory_of $B)" \
            --recordings $RECS --output "$OUT/$B" || echo "FAILED $B"
    done
    echo "SPECIALIST DRENDS LANE $LANE DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=1:mode=shared" -J "argos_specdr_$LANE" "$SELF $LANE --node"
