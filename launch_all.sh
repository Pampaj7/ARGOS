#!/bin/bash
source .miniconda/etc/profile.d/conda.sh
conda activate argos
export PYTHONPATH="$(pwd)"

echo "Starting training on GPU 0,1..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/temporal_refinement/train_warp_fusion.py \
  --out-dir results/03_temporal_refinement/training/adaptive_motion_fusion/exp_200ep_2gpu \
  --epochs 200 \
  --batch-size 4 \
  --sequence-length 16 \
  --bptt-steps 8 \
  --lr 1e-4 \
  --residual-clamp-px 0.5 \
  --sav-weight 0.10 \
  --spatial-weight 0.35 \
  --raw-fidelity-weight 0.35 \
  --motion-comp-weight 0.10 \
  --residual-l1-weight 0.15 \
  --edge-weight 0.04 \
  --alpha-prior-weight 0.50 \
  --alpha-prior-decay-epochs 60 \
  >> results/03_temporal_refinement/training/adaptive_motion_fusion/exp_200ep_2gpu/run.log 2>&1 &
TRAIN_PID=$!

echo "Starting evaluation..."
python3 scripts/temporal_refinement/eval_scripts/evaluate_all_methods_fair_v2.py > results/02_video_stereo/all_methods_fair_eval_v2.log 2>&1 &
EVAL_PID=$!

echo "Both processes started. Waiting for them to finish..."
wait $TRAIN_PID
wait $EVAL_PID
echo "All done!"
