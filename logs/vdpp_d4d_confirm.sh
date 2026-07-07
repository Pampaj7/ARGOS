#!/bin/bash
#BSUB -J vdpp_d4dc
#BSUB -q p1
#BSUB -gpu "num=1:mode=shared"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -W 4:00
#BSUB -o /dtu/p1/leopam/ARGOS/logs/vdpp_d4dc.out
#BSUB -e /dtu/p1/leopam/ARGOS/logs/vdpp_d4dc.err
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
R=results/03_temporal_refinement/vdpp_style_causal_confirmation
for TM in full_history current_frame_only shuffled_history; do
  for SEED in 0 1 2; do
    CK=$R/runs/${TM}__spatial_plus_tgm__lam1.0__clip8__seed${SEED}/best.pt
    [ -f "$CK" ] || { echo "missing $CK"; continue; }
    python scripts/temporal_refinement/vdpp_style_causal/eval_vdpp_d4d.py \
      --ckpt "$CK" --temporal-mode $TM --clips-per-specimen 3 --max-frames 100 \
      --out $R/d4d/${TM}__seed${SEED} 2>&1 | grep -viE 'warn|future' | tail -1
  done
done
echo "VDPP_D4DC_DONE"
