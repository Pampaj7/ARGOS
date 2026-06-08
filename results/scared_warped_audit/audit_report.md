# SCARED Warped Audit

Converted frames: 100 / requested 100
Datasets: dataset_1, dataset_5, dataset_8
Keyframes: dataset_1/keyframe_1, dataset_1/keyframe_2, dataset_1/keyframe_3, dataset_5/keyframe_1, dataset_5/keyframe_2, dataset_5/keyframe_3, dataset_8/keyframe_0, dataset_8/keyframe_1, dataset_8/keyframe_2
Mean valid pixel ratio: 0.700
Median valid pixel ratio: 0.719
Mean depth median: 66.781 mm
Mean disparity median: 78.470 px
Suspicious frames: 0

Outputs:
- Metadata: `results/scared_warped_metadata.csv`
- Montage: `results/scared_warped_audit/montage_50_random_samples.png`
- Valid-ratio histogram: `results/scared_warped_audit/valid_pixel_ratio_histogram.png`

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
