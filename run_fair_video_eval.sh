#!/bin/bash
set -e

source .miniconda/etc/profile.d/conda.sh
conda activate argos
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=1

SCARED_DIR="dataset/SCARED/curated/temporal_gt"
SEQ="$SCARED_DIR/test_dataset_9_keyframe_3"
OUT="results/02_video_stereo/fair_video_eval"
PREDICTIONS_DIR="$OUT/predictions"
mkdir -p $OUT/predictions

# echo "--- 1. Running PPMStereo SF ---"
# python scripts/temporal_refinement/adapters/run_ppmstereo_temporal.py \
#     --sequence-dir $SEQ \
#     --out-dir $OUT/predictions/PPMStereo_SF \
#     --checkpoint external/video_stereo_repos/PPMStereo/ckpt/ppmstereo_weights/ppmstereo_sf.pth \
#     --chunk-size 20

# echo "--- 2. Running PPMStereo DR_SF ---"
# python scripts/temporal_refinement/adapters/run_ppmstereo_temporal.py \
#     --sequence-dir $SEQ \
#     --out-dir $OUT/predictions/PPMStereo_DR_SF \
#     --checkpoint external/video_stereo_repos/PPMStereo/ckpt/ppmstereo_weights/ppmstereo_dr_sf.pth \
#     --chunk-size 20

echo "--- 3. Running TC-Stereo IdentityPose ---"
python scripts/temporal_refinement/adapters/run_tcsm_temporal.py \
    --sequence-dir $SEQ \
    --out-dir $OUT/predictions/TC-Stereo-IdentityPose \
    --checkpoint external/video_stereo_repos/Temporally-Consistent-Stereo-Matching/checkpoints/sceneflow.pth

echo "--- 4. Running evaluate_scared_temporal_gt.py (evaluates SAV, S2M2, and all above!) ---"
python scripts/temporal_refinement/eval_scripts/evaluate_scared_temporal_gt.py \
    --sequence-root $SEQ \
    --out-dir $OUT \
    --device cuda \
    --skip-refiners

echo "Evaluation complete! Check $OUT/temporal_gt_summary.csv"
