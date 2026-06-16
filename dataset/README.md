# ARGOS Dataset Directory

This folder is the local data home for ARGOS.

The rule is simple: one top-level folder per dataset. Raw files, curated subsets, workspace extracts, metadata, and download logs for a dataset live inside that dataset's folder.

```text
dataset/
  SCARED/
  SERVCT/
  StereoMIS/
  D4D/
  EndoSLAM/
  README.md
  manifest.json
```

Large payloads are intentionally ignored by Git. Git should track only lightweight documentation, manifests, and scripts.

## Current Layout

| Dataset | Path | Status | Main Use |
|---|---|---|---|
| SCARED | `SCARED/` | raw archives and curated clips available | surgical stereo GT keyframes and temporal clips |
| SERV-CT | `SERVCT/` | raw source and ARGOS-format samples available | metric SERV-CT benchmark and fine-tuning |
| StereoMIS | `StereoMIS/` | downloaded and inventoried | real surgical stereo-video temporal robustness |
| D4D / Dresden | `D4D/` | metadata downloaded; `specimen_1.tar.gz` downloading | surgical stereo/depth geometry validation |
| EndoSLAM | `EndoSLAM/` | local support snapshot available | future pose/3D validation and pseudo-labeling |

## SCARED

```text
SCARED/
  raw/source/
  curated/consecutive32/
  curated/rect5/
  curated/keyframes_gt_dataset8/
  workspace/scared_consecutive/
```

Important subsets:

- `SCARED/curated/consecutive32/`: 32 consecutive stereo frames from `test_dataset_9/keyframe_3/rgb.mp4`.
- `SCARED/curated/rect5/`: 5 SCARED stereo pairs used for smoke tests.
- `SCARED/curated/keyframes_gt_dataset8/`: SCARED dataset_8 keyframes with GT depth/disparity conversion used by S2M2 benchmarks.
- `SCARED/raw/source/`: raw SCARED archives and extracted sources.

The consecutive clip source video is top/bottom stereo at `1280x2048`; ARGOS splits it into `1280x1024` left/right frames.

## SERV-CT

```text
SERVCT/
  raw/source/
  argos/servct_argos/
  workspace/servct/
```

`SERVCT/argos/servct_argos/` is the canonical ARGOS-format SERV-CT subset.

Each sample contains:

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

## StereoMIS

```text
StereoMIS/
  raw/source/
```

StereoMIS is downloaded and inspected.

Key local files:

- `StereoMIS/raw/source/StereoMIS_0_0_1.zip`
- `StereoMIS/raw/source/INVENTORY.md`
- `StereoMIS/raw/source/stereomis_sequence_inventory.csv`
- `StereoMIS/raw/source/metadata_extract/`
- `StereoMIS/raw/source/preview_extract/`

Confirmed contents:

- 11 vertically stacked stereo videos.
- 90,912 mask PNG files.
- 11 stereo calibration files.
- 11 pose/kinematics `groundtruth.txt` files.
- No dense depth/disparity GT found in archive listing.

Use StereoMIS for temporal robustness and qualitative surgical-video evaluation, not metric depth MAE.

## D4D / Dresden

```text
D4D/
  raw/source/
```

D4D is a high-priority surgical geometry dataset. It is much larger than StereoMIS.

Local status:

- `D4D/raw/source/info.tar.gz` downloaded.
- `D4D/raw/source/info_extract/` extracted.
- `D4D/raw/source/d4d_download_urls.tsv` records OPARA download URLs.
- `D4D/raw/source/specimen_1.tar.gz` is downloading in tmux session `argos_d4d_specimen1_download`.

Monitor D4D download:

```bash
tail -f dataset/D4D/raw/source/d4d_specimen_1_download.log
ls -lh dataset/D4D/raw/source/specimen_1.tar.gz
tmux attach -t argos_d4d_specimen1_download
```

Expected D4D contents after extraction:

- rectified left/right endoscope frames;
- stereo depth maps in metres;
- structured-light point clouds;
- masks;
- camera calibration;
- curated poses.

Convert depth to millimetres for ARGOS reports.

## EndoSLAM

```text
EndoSLAM/
  raw/source/
```

EndoSLAM is currently support data for future domain expansion, pose/3D checks, and possible pseudo-labeling.

## Manifest

Machine-readable dataset metadata is stored in:

```text
manifest.json
```

Update `manifest.json` whenever a dataset is moved, added, downloaded, or converted.
