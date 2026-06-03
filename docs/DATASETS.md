# ARGOS Datasets

ARGOS uses surgical stereo datasets with ground-truth disparity, depth, camera calibration, or enough geometry to derive them.

## Dataset Status

| Dataset | Status | Ground Truth | Current Use |
|---|---|---|---|
| SERV-CT | available | disparity + depth from CT/RGB reference | current benchmark and S2M2 fine-tuning |
| SCARED | downloading | stereo + depth/geometry data | planned large surgical training and cross-dataset validation |
| EndoSLAM | queued | pose/geometry depending on sequence | support data, possible pseudo-labeling/validation |

## Target Unified Format

Converters should emit samples in this structure:

```text
argos_data/<dataset>/<split>/<sample_id>/
  left.png
  right.png
  disp_gt.npy
  depth_gt_mm.npy
  valid_mask.npy
  calib.json
  metadata.json
```

Required metadata:

- `dataset`
- `split`
- `sequence`
- `frame`
- `reference_type`
- `left_path_original`
- `right_path_original`
- `has_disparity_gt`
- `has_depth_gt`
- `units`

Required calibration fields:

- `fx`
- `fy`
- `cx_left`
- `cy_left`
- `cx_right`
- `cy_right`
- `baseline_mm`
- `width`
- `height`

## Split Rules

SERV-CT:

- `zero_shot_eval`: Experiment_1 + Experiment_2
- `honest_train`: Experiment_1
- `honest_test`: Experiment_2
- `all_surgical`: Experiment_1 + Experiment_2

SCARED:

- pending full download and conversion.
- split should avoid mixing frames from the same scene/keyframe family between train and test.

## Notes

- Keep raw datasets outside ARGOS, currently under `/home/pampaj/Desktop/stereo/`.
- Track converter scripts, not converted bulky outputs.
- Converted outputs should be ignored by git unless a tiny smoke fixture is explicitly added.
