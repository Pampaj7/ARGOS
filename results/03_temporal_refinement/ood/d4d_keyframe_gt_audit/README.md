# D4D keyframe stereo GT — Phase 1 forensic audit

Read-only audit of D4D formats + the decisive Zivid→stereo-left transform-chain resolution.
Basis for the sparse-keyframe GT pipeline (`scripts/temporal_refinement/ood/d4d/`).

## Regenerate
```bash
python scripts/temporal_refinement/ood/d4d/inspect_d4d_formats.py   # audit CSVs/JSON
```

## Findings (exact formats)

- **Depth** (`depth_images/*.npy`): float32, 2448×2048 (Zivid colour res), **metres**,
  optical-axis Z, **NaN = invalid**, ~39 % finite (textured surface only). Registered to the
  Zivid colour camera; backprojection with colour K reproduces the `.ply` exactly.
  → `depth_npy_format.csv`
- **Point cloud** (`pointcloud/*.ply`): binary_little_endian, `double x,y,z + uchar r,g,b`,
  ~1.97 M vertices, metres, `zivid_optical_frame`. → `pointcloud_format.csv`
- **SNR** (`snr_images/*.npy`): float32, pixel-aligned to depth, 0–~212 (Zivid signal).
- **Calibration**: `left/right.yaml` (raw K,D,R,P, plumb_bob) + `left/right_rect.yaml`
  (rectified). Stereo rectified **fx = 798.32 px, baseline = 4.235 mm** (from P matrices).
  Zivid colour K: fx ≈ 2486, 2448×2048. Rectification reproducible with OpenCV.
  → `calibration_fields.csv`
- **TF** (`tf/*.json`): `{timestamp, parent_frame, child_frame, transform{translation xyz m,
  rotation xyzw}}`, child→parent (ROS). 5 publishers; endoscope tracked in `polaris_spectra`,
  Zivid in `polaris`, bridged by the **MiRe45** marker. → `tf_format.csv`
- **Clips** (`clips.json`): each clip has `start`/`end` stereo frames + `start_geometry`/
  `end_geometry` → Zivid `.ply` (stem shared with depth `.npy`). 16 sessions →
  **86 clip-anchors, 85 with depth**. → `clip_geometry_mapping.csv`
- **Timestamps**: stereo ~15 fps; TF ~20–30 Hz; per anchor nearest stereo 2–9 ms and pose
  interpolation < 10 ms from the Zivid scan. → `timestamp_analysis.csv`

## Decisive result

The **Zivid→rectified-left transform chain is resolved and empirically validated** (MiRe45
bridge). Full derivation, rejected hypotheses, and the alignment evidence are in
`transform_chain_hypotheses.md`; residual caveats in `blockers.md`.

**D4D can be turned into a valid SPARSE-keyframe stereo benchmark** — not a dense per-frame
one (Zivid is ~2 scans/clip; the scene is non-rigid).

## Files
`dataset_structure.json`, `depth_npy_format.csv`, `pointcloud_format.csv`,
`calibration_fields.csv`, `tf_format.csv`, `timestamp_analysis.csv`,
`clip_geometry_mapping.csv`, `transform_chain_hypotheses.md`, `blockers.md`,
`diagnostics/` (OV_bridge.png, CHAIN_sidebyside.png).
