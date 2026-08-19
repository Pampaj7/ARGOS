#!/bin/bash
# Re-drive the overview figure's five-frame window on the shipped head.
#
# The fused row of Figure 1(a) was produced by the 142-channel configuration, so the paper's
# opening picture showed its ablation. The panels themselves will barely move -- the two
# disparity rows are near-identical by construction, which is the point they make -- but the
# caption quotes the mean |fused - raw| over the window, and that number belongs to whichever
# head produced it.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    cd "$ROOT" || exit 1
    cp /dtu/p1/leopam/ARGOS/ARGOS_hand/paper/figure_assets/fused_window.npz \
       /dtu/p1/leopam/ARGOS/ARGOS_hand/paper/figure_assets/fused_window.142ch.npz 2>/dev/null
    "$PY" scripts/_dump_fused_window.py || { echo "DUMP FAILED"; exit 1; }
    "$PY" scripts/export_figure_panels.py --verify
    echo "FIGURE WINDOW DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 2 -R "span[hosts=1] rusage[mem=10GB]" \
     -gpu "num=1:mode=shared" -J argos_figwin "$SELF --node $*"
