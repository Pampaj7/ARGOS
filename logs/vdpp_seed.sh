#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
OUT=results/03_temporal_refinement/vdpp_style_causal_confirmation/runs
SEED=$1
run(){ RID="$1__$2__lam$3__clip8__seed$SEED"; [ -f "$OUT/$RID/config.json" ] && { echo "skip $RID"; return; }
  python scripts/temporal_refinement/vdpp_style_causal/train_vdpp_causal.py --temporal-input-mode $1 --loss-mode $2 --seed $SEED --lam-tgm $3 --clip-len 8 --steps 1200 --eval-every 1200 --out $OUT 2>&1 | grep -viE 'warn|future'|tail -1; }
run full_history spatial_only 1.0
run full_history spatial_plus_tgm 1.0
run current_frame_only spatial_plus_tgm 1.0
run shuffled_history spatial_plus_tgm 1.0
echo "SEED${SEED}_DONE"
