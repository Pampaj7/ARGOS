#!/bin/bash
# Export the five DRENDS recordings as RGB seed boundaries for BiDAStabilizer.
#
# First half of the out-of-domain head-to-head. The published comparison ran on SCARED-C
# D2, which is the split our checkpoint was selected on while BiDAStabilizer had never
# seen the domain; the paper now declares that and reads the margin as an upper bound.
# On DRENDS neither method has seen the domain.
#
# The grid is 144x180, matching export_scared_d2_smoke.py exactly. That is low for a
# video stabiliser, but it is the grid the published D2 comparison already used, so both
# methods are handicapped identically and the two comparisons stay commensurable.
#
# Throughput work, not timing: GPU contention makes this slower but not wrong. The
# runtime measurement is the one that needs a quiet device and is not run here.
set -u
ROOT=/dtu/p1/leopam/ARGOS
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
    cd "$ROOT/ARGOS_hand/external_comparison" || exit 1
    exec "$ROOT/.miniconda/envs/argos/bin/python" export_drends_boundary.py --device cuda:0 "$@"
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=1:mode=shared" -J argos_drendsexport "$SELF --node $*"
