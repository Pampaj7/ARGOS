#!/bin/bash
# One shard of the A2 closure: a named subset of the six learned-head methods.
#
# The six methods are independent runs over the same frames, so they parallelise perfectly.
# The single-job version was producing one cell every eight minutes and would have taken
# about ten hours for seed 0 alone, with H=4 -- the one row the paper's argument actually
# rests on -- last in a queue behind three horizons nobody is waiting for.
#
# Memory is deliberately small. rusage[mem] is PER CORE, so the 40GB these launchers used
# to ask for reserved 160GB each and three of our own jobs had left 7GB free of 529GB on
# the node: the earlier hour of PEND was self-inflicted, not contention from other users.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
HEAD=A2_no_learned_evidence
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    NAME=$1; shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES shard=$NAME methods=$*"
    exec "$PY" -m model_design.comparison.promoted_head_closure \
        --head "$HEAD" --device cuda:0 --shard "$NAME" --methods "$@"
fi
NAME=$1; shift
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=8GB]" \
     -gpu "num=1:mode=shared" -J "argos_shard_$NAME" "$SELF --node $NAME $*"
