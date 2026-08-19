#!/bin/bash
# TC-Stereo with its temporal path enabled, on SCARED-C's corrected per-frame poses.
#
# The frame-path row already in the paper was justified by the false claim that SCARED-C
# ships no per-frame pose. It does. This runs the competitor the way its authors built it:
# state carried across the sequence and the previous disparity reprojected from pose.
#
# The GPU is not optional even for the pose-convention self-check: softsplat has no CPU
# kernel, it asserts False. So the check runs here, before scoring, and a wrong convention
# aborts the run instead of quietly producing a plausible number.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    # softsplat JIT-compiles its kernel through cupy, which asks cupy for CUDA_HOME only
    # when the variable is unset -- and cupy returns None here, because /usr/local/cuda on
    # this node carries runtime libraries with no headers and no nvcc. CUDA_HOME is used
    # solely as an -I path, so pointing it at the conda prefix skips that lookup entirely.
    export CUDA_HOME=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    cd "$ROOT" || exit 1
    "$PY" scripts/run_tcstereo_temporal.py --device cuda:0 "$@" || { echo "TCSTEREO TEMPORAL FAILED"; exit 1; }
    echo "TCSTEREO TEMPORAL DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 2 -R "span[hosts=1] rusage[mem=10GB]" \
     -gpu "num=1:mode=shared" -J argos_tctemp "$SELF --node $*"
