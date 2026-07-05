# Format notes — physical conventions per dataset

All numbers below are read directly from the data (see `discover_ood_datasets.py`).
These conventions drive the adapters/validation in Phase 2. **Getting any of these
wrong silently produces meaningless OOD metrics**, so each is stated explicitly.

## ARGOS refiner input contract (the target convention)

- Upstream model: **S2M2-S @ 512**, pretrained (zero-shot).
- Raw disparity stored per frame as **float16 `.npy`**, in **original image disparity
  coordinates** (positive pixels, left camera reference, rectified).
  Source: `scripts/temporal_refinement/cache_builders/build_large_v3_s2m2s512_full_cache.py`
  (`"coordinate_system": "original image disparity coordinates"`).
- Refiners consume a **5-frame window** (t-2..t+2) of that disparity + left RGB, expanded
  to the model's 16-channel input by the dataset/feature code. Feature construction to be
  mirrored 1:1 for OOD in Phase 3 (no re-derivation of the channel layout).
- **OOD disparity must be produced by the same pretrained S2M2-S** and left in the same
  coordinates. No rescaling to a canonical baseline, no normalization differences.

## SERV-CT (`dataset/SERVCT/argos/servct_argos/`)

Per-frame directory: `left.png`, `right.png`, `disp_gt.npy`, `depth_gt_mm.npy`,
`valid_mask.npy`, `calib.json`, `metadata.json`.

- **Resolution**: 720 × 576 (W × H), rectified.
- **Disparity GT**: `disp_gt.npy`, float32, **positive pixels**, left reference. Example
  frame `Experiment_2/009`: range 12.86 – 78.84 px, 100% finite. Matches ARGOS sign
  convention → **no sign flip, no rescale**.
- **Depth GT**: `depth_gt_mm.npy`, float32, millimetres. Consistent with
  `depth = fx·baseline / disp` using `calib.json`:
  - `fx = 934.69 px`, `baseline = 5.504 mm` → `fx·baseline ≈ 5145 mm·px`.
  - Round-trip check to be enforced in validation (Phase 2 check #4).
- **valid_mask.npy**: uint8 {0,1}, 1 = valid GT. (Some frames report 100% valid; do not
  assume — mask is read per frame.)
- **Calibration**: `calib.json` gives `fx, fy, cx_left, cy_left, cx_right, cy_right,
  baseline_mm, width, height`. cx_left == cx_right (rectified).
- **Left/right reference**: GT and disparity are aligned to the **left** rectified image.
- **Temporal**: frames within an Experiment are ordered by numeric id but are **sparse**
  (not smooth video). Continuity flag = `weak_sparse` in the manifest.
- **Units field** in metadata: `depth_mm_disparity_px`.
- Raw source retained at `dataset/SERVCT/raw/extracted/SERV-CT/` (untouched).

## D4D (`dataset/D4D/raw/extracted/`)

Per session (e.g. `specimen_1/specimen_1/2025_03_06-16_49_40/`):
`left_images/`, `right_images/`, `color_images/`, `depth_images/`, `snr_images/`,
`camera_info/`, `clips/`, `clips.json`, `masks/`, `pointcloud/`, `tf/`.

- **Stereo images**: `left_images/*.png`, `right_images/*.png` — **unrectified**. Counts
  can differ slightly between sides (e.g. 521 vs 516) → pair by timestamp/id, drop unpaired.
- **Resolution**: ~894 × 714 (varies; read per session).
- **Calibration**: `camera_info/{left,right}.yaml` (raw) and `{left,right}_rect.yaml`
  (rectified P matrices). Rectified left P: `fx ≈ 798.32`, `cx ≈ 435.70`. Stereo baseline
  from `baseline = -P_right[0,3] / fx`. Rectification (`R`,`P`) present but **not applied
  to the stored images** → adapter must `cv2.initUndistortRectifyMap` + `remap`.
- **GT**: **Zivid structured-light** only — `depth_images/` (~2 per session) and
  `pointcloud/` (~2 per session), i.e. GT at scan timepoints, **not per frame**. There is
  **no dense per-frame disparity GT**. `tf/` holds transforms (thousands of entries) for
  pose; `clips.json` defines temporal clips with start/end geometry pointers.
- **Left/right reference**: after rectification, disparity is left-reference; GT depth
  would need reprojection into each left frame using pose from `tf/`.
- **Temporal**: true video within clips (ordered, near-constant fps) → `strong_video`.
- **Loader**: `external/d4d/d4d/loader.py` exposes `D4D → Specimen → Session → Clip` with
  `get_stereo_parameters()`, `load_images(rectified=True)`, `load_depth_maps()`,
  `load_masks()`, pose helpers. Reuse it in the D4D adapter rather than re-parsing.

## Disparity ↔ depth conversions (to be applied in adapters, never tuned per-dataset)

```
depth_mm  = fx_px * baseline_mm / disp_px          # disp>0
disp_px   = fx_px * baseline_mm / depth_mm          # depth>0
```
On resize by factor s (W→s·W): `disp_scaled = s * disp` and `fx_scaled = s * fx`
(baseline unchanged). S2M2-S is run at its native/512 setting and the raw disparity is
kept in that same coordinate space as the GT (resize GT→pred space or pred→GT space
consistently; validation check #5 enforces scale correctness).
