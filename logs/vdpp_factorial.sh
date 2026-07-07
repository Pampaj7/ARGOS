#!/bin/bash
#BSUB -J vdpp_fact
#BSUB -q p1
#BSUB -gpu "num=1:mode=shared"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -W 4:00
#BSUB -o /dtu/p1/leopam/ARGOS/logs/vdpp_factorial.out
#BSUB -e /dtu/p1/leopam/ARGOS/logs/vdpp_factorial.err
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
OUT=results/03_temporal_refinement/vdpp_style_causal_confirmation/runs
run(){ # tmode loss seed lam
  RID="$1__$2__lam$4__clip8__seed$3"
  [ -f "$OUT/$RID/config.json" ] && { echo "skip $RID"; return; }
  python scripts/temporal_refinement/vdpp_style_causal/train_vdpp_causal.py \
    --temporal-input-mode $1 --loss-mode $2 --seed $3 --lam-tgm $4 \
    --clip-len 8 --steps 1500 --eval-every 1500 --out $OUT 2>&1 | grep -viE 'warn|future' | tail -1
}
for SEED in 0 1 2; do
  run full_history       spatial_only     $SEED 1.0
  run full_history       spatial_plus_tgm $SEED 1.0
  run current_frame_only spatial_plus_tgm $SEED 1.0
  run shuffled_history   spatial_plus_tgm $SEED 1.0
done
# tgm-weight sweep on full_history (0.2, 0.5); 1.0 already above
for SEED in 0 1 2; do
  run full_history spatial_plus_tgm $SEED 0.2
  run full_history spatial_plus_tgm $SEED 0.5
done
echo "VDPP_FACTORIAL_DONE"
