# ARGOS Datasets

ARGOS uses surgical stereo datasets with ground-truth disparity, depth, camera calibration, or enough geometry to derive them.

## Dataset Status

| Dataset | Status | Ground Truth | Current Use |
|---|---|---|---|
| SERV-CT | available under `dataset/SERVCT/` | disparity + depth from CT/RGB reference | current benchmark and S2M2 fine-tuning; see `dataset/SERVCT/DATASET_CARD.md` |
| SCARED | reorganized under `dataset/SCARED/curated/`: `geometric_gt/strong_keyframes/` (45 real structured-light keyframes) + `temporal_sequences/` (real video, no GT) | 45 keyframes with real structured-light depth/disparity; temporal video has no per-frame GT | frame-wise geometric benchmark (keyframes) + temporal-consistency/video-stereo work (sequences); see `dataset/SCARED/DATASET_CARD.md` |
| SCARED-C | corrected version under `dataset/SCARED-C/curated/`: same `strong_keyframes/` (25, datasets 1/2/3/6/7 only) + `corrected_temporal_gt/` (17/19 sequences passing a photometric quality gate, 16,921 frames) | COLMAP+scale-recovery reprojected depth/disparity for non-keyframe video frames (pseudo-GT, one tier below structured-light) | main lever for scaling SCARED training data ~100x+; see `dataset/SCARED-C/DATASET_CARD.md` |
| StereoMIS | pilot converted: `dataset/StereoMIS/curated/geometric_gt/temporal_sequences/` (P1, P2_8, P3 — 3/11 sequences, 38,241 rectified frame pairs, 24GB) | pose/kinematics + masks; no dense depth/disparity GT (confirmed against paper) | temporal/pose surgical-video evaluation; see `dataset/StereoMIS/DATASET_CARD.md` |
| D4D / Dresden Dataset | 4/6 specimens extracted + converted (specimen_5 blocked on OPARA outage); 362 anchors, curated-pose GT under `dataset/D4D/processed/keyframe_stereo_gt_curated/` | Zivid structured-light depth/disparity at ~2 keyframes/clip (sparse, not dense) | high-priority geometry + temporal surgical validation; see `dataset/D4D/DATASET_CARD.md` |
| EndoSLAM | queued | pose/geometry depending on sequence | support data, possible pseudo-labeling/validation |

## Local Dataset Layout

ARGOS keeps all local data under `dataset/`. The layout is one top-level folder per dataset.

| Subset | Format | GT | Purpose |
|---|---|---|---|
| `dataset/SCARED/` | `raw/extracted/` (untouched vendor data) + `curated/geometric_gt/strong_keyframes/` + `curated/temporal_sequences/` | yes for strong_keyframes; no GT for temporal_sequences | SCARED metric keyframes (benchmark) and temporal clips (video-stereo work). |
| `dataset/SCARED-C/` | `raw/` (HF download) + `curated/geometric_gt/{strong_keyframes,corrected_temporal_gt}/` | yes (structured-light for keyframes; COLMAP-reprojected pseudo-GT for corrected_temporal_gt) | Scaled-up SCARED training/eval; see `dataset/SCARED-C/DATASET_CARD.md`. |
| `dataset/SERVCT/` | raw source and ARGOS-format samples | yes | SERV-CT baseline and scoreboard evaluation. |
| `dataset/StereoMIS/` | `raw/` (full extraction) + `curated/geometric_gt/temporal_sequences/` (3/11 seqs rectified) | pose/calib/masks; no dense depth found | Real stereo surgical video temporal robustness; see `dataset/StereoMIS/DATASET_CARD.md`. |
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

- fully downloaded and reorganized. `curated/geometric_gt/strong_keyframes/` holds 45 real
  structured-light keyframes (dataset_1-9, vendor-inconsistent numbering: dataset_1-7 use
  keyframe_1..5, dataset_8/9 use keyframe_0..4); `curated/temporal_sequences/` holds real
  stereo video with no per-frame GT (up to 130 frames per dataset_N_keyframe_M sequence).
  Full detail in `dataset/SCARED/DATASET_CARD.md`.
- a rectification bug (`cv2.remap`-ing the depth channel like an image instead of rotating
  points by R1 and z-buffer projecting) was found and fixed in the keyframe eval scripts —
  see `scripts/scared/build_strong_keyframes_rectified.py` for the corrected convention.
- split should avoid mixing frames from the same scene/keyframe family between train and test.

SCARED-C:

- corrected version of SCARED (Han et al., arXiv:2605.16628): replaces kinematics-based
  non-keyframe poses with COLMAP + scale-recovery, extending usable RGB-D from 35 keyframes
  (across SCARED+SCARED-C) to 16,921 corrected temporal frames. Datasets 4/5 excluded (known
  bad calibration, same as vanilla SCARED).
- **not all sequences are trustworthy as-shipped** — a model-free photometric warp-consistency
  gate (`scripts/scared_c/build_quality_gate.py`) rejected 2 of 19 real video sequences
  (bad per-sequence scale-recovery / too-low COLMAP registration overlap); only the 17 passing
  sequences are in `curated/geometric_gt/corrected_temporal_gt/`.
- always mask per-frame with `gt/*_valid.png` — sequence-level "pass" is not a per-frame
  coverage guarantee (~2% of curated frames have zero valid GT pixels).
- full detail, per-sequence gate results, and reproduce commands in
  `dataset/SCARED-C/DATASET_CARD.md`.

StereoMIS:

- real da Vinci Xi stereo endoscopic video (Hayoz et al., arXiv:2304.08023). Zenodo:
  `https://zenodo.org/records/8154924` (raw archive fully downloaded and extracted, 23GB).
- 3 in-vivo porcine subjects, 11 sequences (P1, P2_0-P2_8, P3), vertically-stacked stereo
  video, per-sequence `StereoCalibration.ini`, dense per-frame kinematics pose
  (`groundtruth.txt`), instrument masks (vendor-inconsistent per-frame coverage).
- **no dense depth/disparity GT** — confirmed against the paper itself (poses come from
  forward kinematics, not structured light). Do not report depth/disparity MAE on this
  dataset; keep results labeled temporal/pose/qualitative, separate from metric-GT tables.
- **pilot converted** (`scripts/stereomis/convert.py`): 3 of 11 sequences (P1, P2_8, P3) —
  rectified via `cv2.stereoRectify` (same convention as SCARED), stored as JPEG-95 (not PNG,
  ~5.5x smaller with no real fidelity loss over the already-HEVC source) to stay lean —
  24GB / 38,241 frame pairs. Remaining 8 sequences (P2_0-P2_7) downloaded but not yet
  converted — same `convert.py --sequences ...` extends them.
- split by procedure/patient/video segment, never by adjacent frames from the same continuous clip.
- likely most useful for temporal refinement stress tests: instruments, specularities, smoke/blood/tissue motion, and long consecutive sequences.
- full detail, per-sequence frame/mask counts, and reproduce commands in
  `dataset/StereoMIS/DATASET_CARD.md`.

D4D / Dresden Dataset:

- candidate dataset DOI: `https://doi.org/10.25532/OPARA-1033`. Data descriptor: Docea et al.,
  "The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes"
  (arXiv:2603.02985).
- loader repository cloned at `external/d4d/`; commit `70d6b94ff6de0511a77889597397b23e893559b0`.
- public loader documentation describes porcine cadaver abdominal scenes captured with da Vinci Xi stereo endoscope and Zivid structured-light camera.
- expected clip-level data includes rectified left/right images, left masks, stereo depth maps in metres, structured-light point clouds, Zivid masks, curated camera poses, and camera info.
- per the paper: **6 specimens** (`specimen_0`-`specimen_5`), 150 raw clips pre-filter, **98 curated clips** retained (49 single + 39 incremental + 10 moved-camera). `specimen_0` has 0 retained clips (fully discarded) — no core archive exists for it, only `specimen_0_ambiguous.tar.gz` (not fetched). Abstract's "98 curated recordings" means clips, not session folders.
- OPARA payload is about 447 GB total; core non-ambiguous specimens are about 422 GB.
- download status: `specimen_1.tar.gz` (33.32 GB), `specimen_2.tar.gz` (75GB), `specimen_3.tar.gz` (83GB), `specimen_4.tar.gz` (137GB) all staged, extracted, and converted (362 anchors total). `specimen_5.tar.gz` unavailable — every download attempt (including OPARA's REST content endpoint, tested against a known-good specimen_1 UUID too) returns a server-side error rather than the ~95GB file, so this is an OPARA-side outage, not a bad link — retry later.
- processed keyframe GT lives at `dataset/D4D/processed/keyframe_stereo_gt_curated/` — 362 anchors, 239 usable, projected using the official `curated_camera_pose_{start,end}.txt` (not the raw Polaris tf chain — that "nominal" variant was tried, found strictly worse, and retired; regenerable on demand via `d4d_keyframe_gt.py --pose-source nominal` if ever needed for paper comparison). Full detail in that dir's `DATASET_CARD.md`.
- first use should be subset-first conversion because the full payload is large.
- split by specimen/session/clip; avoid frame-level leakage.
- likely high-value for ARGOS metric validation, but reports must distinguish stereo-derived depth from structured-light/reference geometry.

## Notes

- Keep raw datasets and giant archives inside each dataset folder, e.g. `dataset/SCARED/raw/source/`.
- Keep dataset payload folders ignored by Git.
- Track converter scripts and curated ARGOS-ready subsets used by reports.
- Do not track model weights or full downloaded datasets in Git.
