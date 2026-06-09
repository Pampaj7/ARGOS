# ARGOS Dataset Subsets

This folder is the local data home for ARGOS.

It contains both:

- small ARGOS-ready subsets used directly by current experiments and reports;
- large raw/source datasets used to produce those subsets.

All data payloads in this folder are intentionally ignored by Git, but they live here locally so the repository structure is clear. Git tracks only this `README.md` and `manifest.json`.

## Contents

| Folder | Contents | Used For |
|---|---|---|
| `scared_consecutive32/` | 32 consecutive rectified stereo video frames split into `left/` and `right/`. | Temporal/video-stereo comparison: Stereo Any Video vs S2M2-S vs Fast-FoundationStereo. |
| `scared_rect5/` | 5 SCARED keyframe stereo pairs split into `left/` and `right/`. | Smoke tests and historical keyframe baseline checks. Not valid for temporal claims. |
| `servct_argos/` | SERV-CT ARGOS-format train/test stereo pairs with depth/disparity GT and valid masks. | Surgical GT baseline evaluation and scoreboards. |
| `raw/surgical_stereo/scared/` | Full local SCARED zip archives plus extracted smoke/test folders. | Raw source data for SCARED conversion and temporal clips. |
| `raw/surgical_stereo/servct/` | Raw SERV-CT archive/extract. | Source data for SERV-CT ARGOS conversion. |
| `raw/external_datasets/EndoSLAM/` | Local EndoSLAM snapshot/support data. | Future domain expansion, pose/3D validation, possible pseudo-labeling. |
| `workspace_argos_data/` | Processed data workspace mirrored from the local stereo lab. | Backward-compatible processed-data area used by scripts through symlinks. |

## SCARED Consecutive Clip

Source:

`raw/surgical_stereo/scared/test_dataset_9.zip`

Extracted video:

`test_dataset_9/keyframe_3/rgb.mp4`

The original video is top/bottom stereo at 1280x2048. ARGOS splits it as:

- top half -> `left/*.png`
- bottom half -> `right/*.png`

Current subset:

- 32 consecutive frames.
- Frame range from the source video: frames 50 through 81.
- No GT depth/disparity is included for this clip.

## SERV-CT ARGOS Format

Each experiment folder contains:

- `left.png`
- `right.png`
- `disp_gt.npy`
- `depth_gt_mm.npy`
- `valid_mask.npy`
- `calib.json`
- `metadata.json`

Current split:

- `honest_train/`: 8 samples.
- `honest_test/`: 8 samples.

## Manifest

Machine-readable dataset metadata is stored in:

`manifest.json`

Update the manifest whenever a new curated subset is added.
