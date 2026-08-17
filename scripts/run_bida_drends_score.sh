#!/bin/bash
# Score Raw / BiDAStabilizer / TETHER on the DRENDS boundaries, common support.
#
# Final piece of the out-of-domain head-to-head. BiDA has already written its own
# RAFT-Stereo robust raw and its refined output; this drives TETHER on that same raw and
# scores all three against DRENDS ground truth on one prediction-independent support.
#
# Memory stays modest: a 160GB reservation left this chain queued behind a loaded node
# earlier tonight with "Job requirements for reserving resource (mem) not satisfied",
# which is the real reason to ask for what the work needs rather than what is permitted.
set -u
ROOT=/dtu/p1/leopam/ARGOS
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT/ARGOS_hand/original_h4" || exit 1
    exec "$ROOT/.miniconda/envs/argos/bin/python" scripts/compare_bidastabilizer_drends.py \
        --device cuda:0 "$@"
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=10GB]" \
     -gpu "num=1:mode=shared" -J argos_bidascore "$SELF --node $*"
