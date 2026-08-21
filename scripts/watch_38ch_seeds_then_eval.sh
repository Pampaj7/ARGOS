#!/bin/bash
set -u
R=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4/model_design/training_runs
while true; do
    ok=1
    for V in A3b_single_resolution_38ch A6_geometry_only A1_no_appearance; do
        for S in 1 2; do
            N=$(( $(wc -l < "$R/ablation_${V}_seed_${S}/training_history.csv" 2>/dev/null || echo 1) - 1 ))
            [ "$N" -ge 12 ] || ok=0
        done
    done
    [ "$ok" = 1 ] && break
    sleep 600
done
sleep 60
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
SELF=$(readlink -f "$0")
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=12GB]" \
     -gpu "num=2:mode=shared" -J argos_38seedeval /bin/bash -c '
BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits \
    | awk -F", " -v busy="$BUSY" "{used=index(busy,\$2)>0; print used, -\$3, \$1}" \
    | sort -k1,1n -k2,2n | head -1 | awk "{print \$3}")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4 || exit 1
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
for F in a3b_seed1 a3b_seed2 a6_seed1 a6_seed2 a1_seed1 a1_seed2; do
    DEST=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/ablation_eval/$F
    [ -d "$DEST/runs/scared-d7" ] && { echo "=== $F already done"; continue; }
    echo "=== $F"
    "$PY" model_design/comparison/run_definitive_evaluation.py --datasets scared-d2 scared-d7 \
        --module "model_design.comparison.ablation_h4:factory_$F" --device cuda:0 \
        --output "$DEST" || echo "FAILED $F"
done
echo "38CH SEED EVALS DONE"'
