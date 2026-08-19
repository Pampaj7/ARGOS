#!/bin/bash
# Stage-by-stage runtime for the canonical head and for A2, on a quiet GPU.
#
# Two things make this worth pinning to a host. Latency measured on a contended GPU is not
# latency, and n-62-12-83 is currently running nine jobs per H100 while n-62-12-84 sits at
# one. Nothing in our launchers ever expressed a host preference, so LSF kept packing 83
# and every previous attempt at this measurement would have produced a number about the
# queue rather than about the module.
#
# The comparison is the point: A2 drops the frozen ResNet-18 entirely -- 157,504 parameters
# and three forward passes per frame -- so if the promotion holds, the overhead it removes
# is measured here rather than asserted.
set -u
ROOT=/dtu/p1/leopam/ARGOS
PY=$ROOT/.miniconda/envs/argos/bin/python
EVAL=$ROOT/ARGOS_hand/original_h4/scripts/measure_runtime.py
OUT=$ROOT/ARGOS_hand/results/runtime
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
    export CUDA_VISIBLE_DEVICES
    echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")"
    cd "$ROOT/ARGOS_hand/original_h4" || exit 1
    # Running the script by absolute path puts scripts/ on sys.path, not the working
    # directory, so `import model_design` fails. The other launchers avoid this by using
    # `python -m` from here; this one needs the package root stated.
    export PYTHONPATH="$ROOT/ARGOS_hand/original_h4${PYTHONPATH:+:$PYTHONPATH}"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | wc -l | xargs echo "other processes on this GPU:"
    echo "=== canonical (142 channels, frozen ResNet-18 in the loop)"
    "$PY" "$EVAL" --device cuda:0 --module model_design.comparison.canonical_h4:factory \
        --output "$OUT/canonical_h4_quiet.json" || echo "FAILED canonical"
    echo "=== A2 (38 channels, no extractor)"
    "$PY" "$EVAL" --device cuda:0 --module model_design.comparison.ablation_h4:factory_a2 \
        --output "$OUT/a2_quiet.json" || echo "FAILED a2"
    echo "RUNTIME PAIR DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
# n-62-12-84 is an H100 host but not in p1i ("Host or host group is not used by the
# queue"), and gpuv100i is at its own job limit, so the quiet-GPU plan reduces to asking
# for an exclusive GPU here and reporting which host we landed on. A contended measurement
# is reported as contended rather than quietly published as latency.
# An exclusive GPU never became available: p1i is one node, both its H100s carry
# other users' work, and the request pended for three hours. Shared it is -- which makes
# the ABSOLUTE numbers unusable as latency, and the RELATIVE one still sound: the two heads
# run back to back in the same job on the same device, so whatever contention exists applies
# to both. What this measures is the cost A2 removes, not what either costs.
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=8GB]" \
     -gpu "num=1:mode=shared" -J argos_runtimepair "$SELF --node $*"
