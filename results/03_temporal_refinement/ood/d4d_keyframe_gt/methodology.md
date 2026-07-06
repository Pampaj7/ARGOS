# Methodology — D4D keyframe stereo GT

## Steps
1. **Forensic audit** (`inspect_d4d_formats.py`): exact depth/ply/tf/calibration/clip formats
   + timestamp coverage. Read-only.
2. **Rectification**: `cv2.initUndistortRectifyMap`/`remap` from `left/right(_rect).yaml`
   (bilinear for images, geometric projection for depth — never image-resize disparity).
3. **Transform chain** (`d4d_keyframe_gt.chain_cam_from_zivid`): 4 tracker poses
   (`polaris_spectra→camera_optical`, `polaris_spectra→MiRe45`, `polaris→MiRe45`,
   `polaris→zivid`) interpolated (lerp + slerp) to the Zivid scan timestamp, composed with the
   MiRe45 marker bridge. Validated by anatomical reprojection (rejected hypotheses documented).
4. **Projection**: backproject Zivid depth (colour K) → 3D points (+SNR) → camera_optical →
   rectified-left (R_left) → pixels (P_left_rect), **z-buffered** (nearest wins).
5. **Disparity**: `fx·baseline/Z`, fx & baseline derived from the rectified P matrices
   (not hard-coded). Round-trip depth↔disparity verified exact.
6. **Quality filtering** (`finalize_d4d_benchmark.py`): transparent thresholds on coverage,
   time offset, interpolation gap, disparity plausibility → valid / usable_with_warning /
   rejected.
7. **Evaluator** (`evaluate_d4d_keyframes.py`): MAE, Bad-1/3/5, depth MAE, boundary/interior,
   SNR-stratified, at anchors; accepts arbitrary predictions.

## Validation performed
- Depth backprojection ≡ point cloud (identical XYZ ranges, same vertex count).
- Transform chain: rejected H0 (no bridge → points behind camera) and H1-inverse (degenerate
  far projection); accepted the MiRe45 bridge by anatomical/instrument alignment.
- Depth↔disparity round-trip: 0.00 px across all 58 anchors.
- Timestamp offsets quantified per anchor.
- Evaluator self-consistency: identity prediction → 0 error on all metrics.

## Explicitly NOT done (no fabricated GT)
- No dense per-frame GT. No rigid propagation of a Zivid scan across a non-rigid clip.
- Local pseudo-GT (±1–3 frames) prepared but not run.
