# ARGOS Datasets

ARGOS uses surgical stereo datasets with ground-truth disparity, depth, camera calibration, or enough geometry to derive them.

## Dataset Status

| Dataset | Status | Ground Truth | Current Use |
|---|---|---|---|
| SERV-CT | available under `dataset/SERVCT/` | disparity + depth from CT/RGB reference | current benchmark and S2M2 fine-tuning |
| SCARED | raw archives and curated clips under `dataset/SCARED/` | stereo + depth/geometry data for full dataset; current temporal clip has no GT | temporal/video-stereo comparison and planned large surgical training |
| StereoMIS | downloaded and inventoried under `dataset/StereoMIS/` | stereo video, calibration, masks, pose/kinematics; no dense depth/disparity GT found | high-value temporal surgical-video domain expansion |
| D4D / Dresden Dataset | loader cloned; metadata downloaded; `specimen_1.tar.gz` download running under `dataset/D4D/` | rectified stereo, stereo depth maps, structured-light point clouds, masks, camera calibration | high-priority geometry + temporal surgical validation |
| EndoSLAM | queued | pose/geometry depending on sequence | support data, possible pseudo-labeling/validation |

## Local Dataset Layout

ARGOS keeps all local data under `dataset/`. The layout is one top-level folder per dataset.

| Subset | Format | GT | Purpose |
|---|---|---|---|
| `dataset/SCARED/` | raw source, curated clips, workspace extracts | mixed | SCARED metric keyframes and temporal clips. |
| `dataset/SERVCT/` | raw source and ARGOS-format samples | yes | SERV-CT baseline and scoreboard evaluation. |
| `dataset/StereoMIS/` | raw archive, metadata extract, inventory, preview | pose/calib/masks; no dense depth found | Real stereo surgical video temporal robustness. |
| `dataset/D4D/` | metadata, download URLs, staged specimen payloads | expected depth/pointcloud/calib | Dresden D4D surgical stereo/depth validation. |
| `dataset/EndoSLAM/` | EndoSLAM support data | mixed | Future domain expansion and pose/3D validation. |

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

StereoMIS:

- downloaded dataset from Zenodo for real da Vinci Xi stereo endoscopic video.
- shared record: `https://zenodo.org/records/7727692`; prefer latest linked version `https://zenodo.org/records/8154924`.
- public description reports 3 in-vivo porcine subjects and 11 surgical sequences with breathing, tool motion, and tissue deformation.
- inspected files include 11 vertically stacked stereo videos, 90,912 masks, 11 stereo calibration files, and 11 pose/kinematics `groundtruth.txt` files.
- no dense depth/disparity GT was found in the archive listing.
- first use should be unsupervised/teacher temporal evaluation.
- split by procedure/patient/video segment, never by adjacent frames from the same continuous clip.
- likely most useful for temporal refinement stress tests: instruments, specularities, smoke/blood/tissue motion, and long consecutive sequences.

D4D / Dresden Dataset:

- candidate dataset DOI: `https://doi.org/10.25532/OPARA-1033`.
- loader repository cloned at `external/d4d/`; commit `70d6b94ff6de0511a77889597397b23e893559b0`.
- public loader documentation describes porcine cadaver abdominal scenes captured with da Vinci Xi stereo endoscope and Zivid structured-light camera.
- expected clip-level data includes rectified left/right images, left masks, stereo depth maps in metres, structured-light point clouds, Zivid masks, curated camera poses, and camera info.
- OPARA payload is about 447 GB total; core non-ambiguous specimens are about 422 GB.
- `specimen_1.tar.gz` is the first staged download, about 33.32 GB.
- first use should be subset-first conversion because the full payload is large.
- split by specimen/session/clip; avoid frame-level leakage.
- likely high-value for ARGOS metric validation, but reports must distinguish stereo-derived depth from structured-light/reference geometry.

## Notes

- Keep raw datasets and giant archives inside each dataset folder, e.g. `dataset/SCARED/raw/source/`.
- Keep dataset payload folders ignored by Git.
- Track converter scripts and curated ARGOS-ready subsets used by reports.
- Do not track model weights or full downloaded datasets in Git.
