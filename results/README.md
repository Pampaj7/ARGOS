# ARGOS Stereo Results

This folder stores ARGOS-owned outputs that are small enough to keep organized: reports, scoreboards, plots, JSON/CSV metrics, smoke logs, and presentation images.

## Current Structure

| Path | Purpose |
|---|---|
| `01_frame_stereo/` | Baseline tables, metrics, and comparisons for pure frame-by-frame models (e.g. S2M2, Fast-FoundationStereo) on SERV-CT and SCARED. |
| `02_video_stereo/` | Scouting reports and comparisons involving temporal baselines (e.g. StereoAnyVideo, TC-Stereo). |
| `03_temporal_refinement/` | Outputs, metrics, and temporal evaluations specific to the ARGOS temporal refinement models (ConvGRU, Tiny U-Net). |
| `04_dataset_derivatives/` | Extracted sequences, caches, and intermediate predictions used for generating results. |
| `images/` | Portable montage/scoreboard PNGs for global use. |

## Convention

New experiment outputs should follow the numbering scheme and be placed in their corresponding domain:

```text
results/<domain>/<experiment_name>/
  report.md
  report.json
  metrics/
  images/
  logs/
```

Do not store raw datasets or large checkpoints here.
