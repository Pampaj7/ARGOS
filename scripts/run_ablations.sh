#!/bin/bash
# Pre-registered architecture ablations, two H100s in one p1i job.
# GPU 0: A4 (relaxed convexity) then A2 (no learned evidence).
# GPU 1: A1 (no appearance channels).
# Declared in model_design/ablation_preregister.json.
set -u
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
LOGS=/dtu/p1/leopam/ARGOS/logs
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--node" ]; then
    shift
    cd "$ROOT" || exit 1
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
    ( CUDA_VISIBLE_DEVICES=0 "$PY" model_design/train_ablation.py --variant A4_relaxed_convexity \
        > "$LOGS/train_A4.log" 2>&1
      CUDA_VISIBLE_DEVICES=0 "$PY" model_design/train_ablation.py --variant A2_no_learned_evidence \
        > "$LOGS/train_A2.log" 2>&1 ) &
    echo "GPU 0 -> A4 then A2 (pid $!)"
    sleep 5
    CUDA_VISIBLE_DEVICES=1 "$PY" model_design/train_ablation.py --variant A1_no_appearance \
        > "$LOGS/train_A1.log" 2>&1 &
    echo "GPU 1 -> A1 (pid $!)"
    wait
    echo "ABLATIONS DONE"
    exit 0
fi
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=40GB]" \
     -gpu "num=2:mode=shared" -J argos_ablate "$SELF --node $*"
