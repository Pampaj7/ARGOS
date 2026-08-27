#!/bin/bash
# Dump the per-frame arrays the DRENDS video reads. One GPU pass, no training, no
# threshold, and nothing written that the paper cites: the evaluator that produces the
# published DRENDS figures is imported, not edited.
#
# All five recordings. Choosing the four that improve most would exclude Vid13, which is
# the recording Sec. V-I already names as the one where our intervention is a net harm on
# BiDA's support -- and a paper that spent this long refusing to select seeds, arenas and
# thresholds should not select recordings for its video.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    export CUDA_VISIBLE_DEVICES=${PIN_GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
    cd "$ROOT" || exit 1
    exec "$PY" scripts/dump_drends_frames.py --backbone RAFT-Stereo --device cuda:0 "$@"
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=1:mode=shared" -J argos_vidfr "$SELF --node $*"
