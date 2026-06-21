#!/bin/bash
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
source .miniconda/etc/profile.d/conda.sh
conda activate argos

echo "Testing CUDA..."
python -c "import torch; print('CUDA IS AVAILABLE:', torch.cuda.is_available())"

mkdir -p results/03_temporal_refinement/training/adaptive_motion_fusion/live_run_v4

echo "Starting training..."
torchrun --nproc_per_node=2 scripts/temporal_refinement/train_adaptive_motion_fusion.py \
    --out-dir results/03_temporal_refinement/training/adaptive_motion_fusion/live_run_v4 \
    --epochs 240 \
    --batch-size 4 \
    --eval-every 5 \
    --save-every 10 \
    --num-workers 8
