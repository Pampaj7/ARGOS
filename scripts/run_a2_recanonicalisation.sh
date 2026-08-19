#!/bin/bash
# Every analysis the paper attributes to TETHER but only ever ran on the 142-channel head.
#
# The recanonicalisation renamed the model and rebuilt Tables 1-4, and stopped there. A
# word-by-word audit found seventeen blocking discrepancies, and nine of them are one rerun
# each with the shipped head substituted for the canonical factory: the branch decomposition,
# the oracle/risk analysis, the BiDA in-domain common support, the D4D degenerate-policy
# diagnostic, and the two figures. Nothing here is a new experiment. Each is an existing
# analysis pointed at the model the paper actually proposes.
#
# Every step writes to its own directory rather than over the 142-channel numbers, because
# the paper still reports that configuration as its ablation and needs both.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
RES=/dtu/p1/leopam/ARGOS/ARGOS_hand/results
A2=model_design.comparison.ablation_h4:factory_a2
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    # Prefer a GPU with no compute process at all, and fall back to most-free-memory. Free
    # memory alone is a poor criterion when jobs dispatch minutes apart: they read the same
    # snapshot and pile onto the same device while the other sits at 0% -- observed twice.
    BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v busy="$BUSY" '{used=index(busy,$2)>0; print used, -$3, $1}' \
        | sort -k1,1n -k2,2n | head -1 | awk '{print $3}')
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    cd "$ROOT" || exit 1

    for B in fusion_only reset_only; do
        if [ -f "$RES/definitive_evaluation/experimental_closure/d2_branch_${B}_tether/summary.csv" ]; then
            echo "=== branch $B already done"; continue; fi
        echo "=== branch $B"
        "$PY" -m model_design.comparison.branch_sweep --branch "$B" --head tether \
              --device cuda:0 || echo "FAILED branch $B"
    done

    if [ ! -f "$RES/oracle_risk_analysis/a2/oracle_risk_metrics.csv" ]; then
        echo "=== oracle and risk"
        "$PY" scripts/analyze_oracle_and_risk.py --module "$A2" --device cuda:0 \
            || echo "FAILED oracle"
    else echo "=== oracle already done"; fi

    if [ ! -f "$RES/bida_common_support_a2/pooled.json" ]; then
        echo "=== bida in-domain common support"
        "$PY" scripts/compare_bidastabilizer.py --module "$A2" --device cuda:0 \
            --output "$RES/bida_common_support_a2" || echo "FAILED bida"
    else echo "=== bida already done"; fi

    if [ ! -f "$RES/d4d_degenerate/ablation_h4_factory_a2/d4d_diagnostics.csv" ]; then
        echo "=== d4d degenerate-policy diagnostic"
        "$PY" -m model_design.comparison.run_definitive_evaluation --datasets d4d \
              --module "$A2" --device cuda:0 || echo "FAILED d4d"
    else echo "=== d4d already done"; fi

    echo "=== teaser panels"
    "$PY" scripts/_dump_fused_window.py || echo "FAILED teaser"
    echo "=== frame scatter"
    "$PY" scripts/plot_frame_scatter.py || echo "FAILED scatter"

    echo "A2 RECANONICALISATION DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=10GB]" \
     -gpu "num=1:mode=shared" -J argos_a2recanon "$SELF --node $*"
