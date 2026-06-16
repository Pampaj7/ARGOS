# SCARED Warped Audit

Converted frames: 1898 / requested 100
Datasets: dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6
Keyframes: dataset_1/keyframe_1, dataset_1/keyframe_2, dataset_1/keyframe_3, dataset_2/keyframe_1, dataset_2/keyframe_2, dataset_2/keyframe_3, dataset_2/keyframe_4, dataset_3/keyframe_1, dataset_3/keyframe_2, dataset_3/keyframe_3, dataset_3/keyframe_4, dataset_4/keyframe_1, dataset_4/keyframe_2, dataset_4/keyframe_3, dataset_4/keyframe_4, dataset_5/keyframe_1, dataset_5/keyframe_2, dataset_5/keyframe_3, dataset_5/keyframe_4, dataset_6/keyframe_1, dataset_6/keyframe_2, dataset_6/keyframe_3, dataset_6/keyframe_4
Mean valid pixel ratio: 0.344
Median valid pixel ratio: 0.338
Mean depth median: nan mm
Mean disparity median: nan px
Suspicious frames: 382

Outputs:
- Metadata: `results/scared_warped_train_subset_metadata.csv`
- Montage: `results/scared_warped_train_subset_audit/montage_50_random_samples.png`
- Valid-ratio histogram: `results/scared_warped_train_subset_audit/valid_pixel_ratio_histogram.png`

Clean keyframe comparison:
{
  "valid_pixel_ratio": {
    "mean": 0.5466278584798177,
    "median": 0.5467140197753906
  },
  "depth_median_mm": {
    "mean": 62.414222632514104,
    "median": 58.56782913208008
  },
  "depth_p95_mm": {
    "mean": 82.96121427747939,
    "median": 76.23619842529297
  },
  "disp_median_px": {
    "mean": 86.15886306762695,
    "median": 84.35633087158203
  },
  "disp_p95_px": {
    "mean": 125.82993689643013,
    "median": 122.51698303222656
  },
  "frames": 45
}

Suspicious frame examples:
- dataset_2/keyframe_2/frame_000260: valid coverage < 20%
- dataset_2/keyframe_2/frame_000270: valid coverage < 20%
- dataset_2/keyframe_2/frame_000280: valid coverage < 20%
- dataset_2/keyframe_2/frame_000290: valid coverage < 20%
- dataset_2/keyframe_2/frame_000300: valid coverage < 20%
- dataset_2/keyframe_2/frame_000310: valid coverage < 20%
- dataset_2/keyframe_2/frame_000320: valid coverage < 20%
- dataset_2/keyframe_2/frame_000330: valid coverage < 20%
- dataset_2/keyframe_2/frame_000340: valid coverage < 20%
- dataset_2/keyframe_2/frame_000350: valid coverage < 20%
- dataset_2/keyframe_2/frame_000360: valid coverage < 20%
- dataset_2/keyframe_2/frame_000370: valid coverage < 20%
- dataset_2/keyframe_2/frame_000380: valid coverage < 20%
- dataset_2/keyframe_2/frame_000390: valid coverage < 20%
- dataset_2/keyframe_2/frame_000400: valid coverage < 20%
- dataset_2/keyframe_2/frame_000410: valid coverage < 20%
- dataset_2/keyframe_2/frame_000570: valid coverage < 20%
- dataset_2/keyframe_2/frame_000580: valid coverage < 20%
- dataset_2/keyframe_2/frame_000590: valid coverage < 20%
- dataset_2/keyframe_2/frame_000600: valid coverage < 20%
