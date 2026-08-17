#!/bin/bash
# Canonical seeds 1 and 2 on DRENDS, three backbones each.
#
# The seed study was run on SCARED only, so the canonical model has exactly one DRENDS
# number (results/drends_masked, seed 0) while A2 now has three. Comparing a three-seed
# mean against a one-seed baseline is the error we already made once on the ablations,
# where seed 0 turned out to be the weakest of the three and flattered every variant.
# This fills in the missing two seeds so the out-of-domain comparison is matched 3 vs 3.
#
# Seed 0 is not re-run: results/drends_masked already holds it under the same protocol.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
OUT=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/drends_canonical_seeds
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$ROOT" || exit 1
    for SEEDSPEC in "factory_seed1:seed1" "factory_seed2:seed2"; do
        F="${SEEDSPEC%%:*}"; S="${SEEDSPEC##*:}"
        for B in RAFT-Stereo CREStereo Fast-FoundationStereo; do
            echo "=== $S / $B"
            "$PY" -m model_design.comparison.drends_backbone_transfer --device cuda:0 \
                --backbone "$B" --module "model_design.comparison.seed_h4:$F" \
                --output "$OUT/$S/$B" || echo "FAILED $S $B"
        done
    done
    echo "CANONICAL DRENDS SEEDS DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=2:mode=shared" -J argos_canondrends "$SELF --node $*"
