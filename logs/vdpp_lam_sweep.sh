#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
OUT=results/03_temporal_refinement/vdpp_style_causal_confirmation/runs
run(){ RID="full_history__spatial_plus_tgm__lam$1__clip8__seed$2"; [ -f "$OUT/$RID/config.json" ] && { echo "skip $RID"; return; }
  python scripts/temporal_refinement/vdpp_style_causal/train_vdpp_causal.py --temporal-input-mode full_history --loss-mode spatial_plus_tgm --seed $2 --lam-tgm $1 --clip-len 8 --steps 1200 --eval-every 1200 --out $OUT 2>&1 | grep -viE 'warn|future'|tail -1; }
for LAM in 0.2 0.5; do
  for SEED in 0 1 2; do
    run $LAM $SEED
  done
done
echo "LAM_SWEEP_DONE"
