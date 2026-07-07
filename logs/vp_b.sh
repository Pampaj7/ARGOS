#!/bin/bash
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos; cd /dtu/p1/leopam/ARGOS; export PYTHONPATH="$(pwd)"
T=results/03_temporal_refinement/vdpp_style_causal_confirmation/runs
r(){ RID="$1__spatial_plus_tgm__lam1.0__clip8__seed$2"; [ -f "$T/$RID/config.json" ] && { echo "skip $RID"; return; }
  python scripts/temporal_refinement/vdpp_style_causal/train_vdpp_causal.py --temporal-input-mode $1 --loss-mode spatial_plus_tgm --seed $2 --lam-tgm 1.0 --clip-len 8 --steps $3 --eval-every $3 --out $T 2>&1|grep -viE 'warn|future'|tail -1; }
r current_frame_only 1 1200
r shuffled_history 2 700
echo "PAIR_DONE"
