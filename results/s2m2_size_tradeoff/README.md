# S2M2 Size Tradeoff On SCARED

Benchmark of S2M2-S, S2M2-L, and S2M2-XL on rectified SCARED dataset_8 clean keyframes with GT depth converted to disparity.

The originally requested converted path, `stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes`, is not present in this workspace. This run uses the current ARGOS clean keyframe subset:

`/home/pampaj/Desktop/ARGOS/dataset/scared_keyframes_gt_dataset8/dataset_8`

## Outputs

- `s2m2_size_tradeoff.csv`: aggregate metrics per model and resize setting.
- `s2m2_size_tradeoff_frame_metrics.csv`: per-frame metrics.
- `s2m2_size_tradeoff.json`: machine-readable benchmark payload.
- `s2m2_size_tradeoff.md`: analysis report and recommendations.
- `qualitative/`: montage PNGs with left image, GT disparity, GT depth, predictions, and absolute error maps.
- `run.log`: stdout/stderr from the latest run.

## Run Command

```bash
cd /home/pampaj/Desktop/ARGOS
PYTHONPATH=/home/pampaj/Desktop/stereo/s2m2/src \
/home/pampaj/Desktop/stereo/Fast-FoundationStereo/.conda/bin/python \
  scripts/scared/benchmark_s2m2_size_tradeoff.py \
  --scared_root /home/pampaj/Desktop/ARGOS/dataset/scared_keyframes_gt_dataset8/dataset_8 \
  --s2m2_src /home/pampaj/Desktop/stereo/s2m2/src \
  --weights_dir /home/pampaj/Desktop/stereo/s2m2/weights/pretrain_weights \
  --out_dir /home/pampaj/Desktop/ARGOS/results/s2m2_size_tradeoff \
  --models S L XL \
  --widths 0 1024 736 512 \
  --refine_iter 3
```

## Current Takeaway

On this 5-keyframe SCARED subset, XL full resolution is the most accurate, but the margin over L is very small: about `0.0136 mm` depth MAE and `0.0142 px` disparity MAE at full resolution. XL is much slower and heavier, so L is a strong default candidate unless future larger SCARED evaluation shows XL gains on hard frames.

