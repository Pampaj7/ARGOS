# ARGOS Datasets

ARGOS uses surgical stereo datasets with ground-truth disparity, depth, camera calibration, or enough geometry to derive them.

## Dataset Status

| Dataset | Status | Ground Truth | Current Use |
|---|---|---|---|
| SERV-CT | available; curated subset mirrored in `dataset/servct_argos/` | disparity + depth from CT/RGB reference | current benchmark and S2M2 fine-tuning |
| SCARED | raw archives available locally; curated clips mirrored in `dataset/scared_*` | stereo + depth/geometry data for full dataset; current clip has no GT | temporal/video-stereo comparison and planned large surgical training |
| EndoSLAM | queued | pose/geometry depending on sequence | support data, possible pseudo-labeling/validation |

## Local Dataset Layout

ARGOS keeps all local data under `dataset/`. Small ready-to-use subsets sit at the top of `dataset/`, while raw/full sources live under `dataset/raw/` and are ignored by Git.

| Subset | Format | GT | Purpose |
|---|---|---|---|
| `dataset/scared_consecutive32/` | `left/*.png`, `right/*.png` | no | 32-frame consecutive temporal stereo comparison. |
| `dataset/scared_rect5/` | `left/*.png`, `right/*.png` | no | 5-keyframe smoke tests; not valid for temporal claims. |
| `dataset/servct_argos/` | ARGOS sample folders with `left/right`, GT, mask, calib, metadata | yes | SERV-CT baseline and scoreboard evaluation. |
| `dataset/raw/surgical_stereo/scared/` | SCARED zip archives and extracted source folders | mixed | Raw source for conversion and clip extraction. |
| `dataset/raw/surgical_stereo/servct/` | SERV-CT zip/archive extract | yes | Raw source for SERV-CT conversion. |
| `dataset/raw/external_datasets/EndoSLAM/` | EndoSLAM support data | mixed | Future domain expansion and pose/3D validation. |
| `dataset/workspace_argos_data/` | Processed local data workspace | mixed | Backward-compatible processed data used by scripts. |

See `dataset/manifest.json` for source paths and exact extraction notes.

## Target Unified Format

Converters should emit samples in this structure:

```text
dataset/<dataset>/<split>/<sample_id>/
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

- Keep raw datasets and giant archives under `dataset/raw/` for local clarity.
- Keep `dataset/raw/` and `dataset/workspace_argos_data/` ignored by Git.
- Track converter scripts and curated ARGOS-ready subsets used by reports.
- Do not track model weights or full downloaded datasets in Git.
