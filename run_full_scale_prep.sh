#!/bin/bash
set -e

source .miniconda/etc/profile.d/conda.sh
conda activate argos
export PYTHONPATH="$(pwd)"

# echo "Step 1: Extracting ALL SCARED source videos..."
# python3 scripts/temporal_refinement/data_prep/extract_scared_full.py

# echo "Step 2: Predicting S2M2-S512 (Fast Backbone)..."
# CUDA_VISIBLE_DEVICES=1 python3 scripts/temporal_refinement/data_prep/predict_s2m2_long_sequences.py \
#     --variant S \
#     --width 512 \
#     --out-root results/04_dataset_derivatives/SCARED/scared_long_predictions/s2m2_s512

# echo "Step 3: Predicting S2M2-L736 (High-Fidelity Spatial Teacher)..."
# CUDA_VISIBLE_DEVICES=1 python3 scripts/temporal_refinement/data_prep/predict_s2m2_long_sequences.py \
#     --variant L \
#     --width 736 \
#     --out-root results/04_dataset_derivatives/SCARED/scared_long_predictions/s2m2_l736

echo "Step 4: Predicting StereoAnyVideo (Geometric/Temporal Teacher)..."
CUDA_VISIBLE_DEVICES=1 python3 scripts/temporal_refinement/data_prep/predict_stereoanyvideo_long_sequences.py \
    --out-root results/04_dataset_derivatives/SCARED/scared_long_predictions/stereoanyvideo_384x640

echo "Step 5: Building V3 Cache for fast training..."
python3 scripts/temporal_refinement/cache_builders/build_large_v3_s2m2s512_full_cache.py

echo "DONE! The full dataset is ready at results/03_temporal_refinement/cache/large_v3_s2m2s512_full"
