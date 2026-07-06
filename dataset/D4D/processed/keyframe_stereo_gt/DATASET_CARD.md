# D4D Keyframe Stereo GT — Dataset Card

## Summary
Sparse-keyframe stereo depth/disparity ground truth derived from the D4D surgical dataset's
**Zivid structured-light** acquisitions, projected into the **rectified stereo-left** frame.
Real GT at the ~2 Zivid scan instants per clip — **not** dense per-frame GT.

## Original D4D
Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes. Hierarchy:
`specimen → session → clip`; per session: unrectified stereo video (`left/right_images`,
~15 fps), rectified calibration, Polaris optical-tracker poses (`tf/`), and ~2 Zivid scans
(`depth_images` metres, `pointcloud` .ply, `snr_images`, `color_images`).

## Processed benchmark
- **Anchors**: 197 built (specimen_1: 82, specimen_2: 39, specimen_3: 76). specimen_4
  extraction in progress; specimen_5 archive corrupt.
- **Quality**: 142 valid + 24 usable_with_warning + 31 rejected → **166 usable**.
- **Per anchor**: `left_rectified.png`, `right_rectified.png` (when the right pair exists),
  `gt_depth_left.npy` (metres, optical-axis Z), `gt_disparity_left.npy` (positive px,
  left reference), `valid_mask.png`, `snr_mask.npy`, `metadata.json`.

## Transform chain (per-specimen convention, validated by reprojection)
Zivid geometry → rectified-left via the Polaris tracker. Three conventions auto-detected
(never forced):
- `mire45_bridge` (specimen_1 subset): camera in `polaris_spectra`, Zivid in `polaris`,
  bridged by the MiRe45 marker: `inv(T_ps←cam)·T_ps←M45·inv(T_pol←M45)·T_pol←zivid`.
- `direct_ps` (specimen_2, specimen_3): both in `polaris_spectra`: `inv(T_ps←cam)·T_ps←zivid`.
- `direct_polaris` (specimen_1/3 subset): both in `polaris`: `inv(T_pol←cam)·T_pol←zivid`.
Each validated by anatomical/instrument alignment of the projected Zivid cloud onto the
endoscope image (`results/.../d4d_keyframe_gt_audit/transform_chain_hypotheses.md`).

## Coordinate conventions & calibration
- Disparity: positive px, **left rectified reference**. Depth: metres, optical-axis Z.
- `disparity = fx·baseline/Z`, fx and baseline from the rectified P matrices (per session).
- **Calibration varies by specimen** (recorded, not forced): e.g. specimen_1 fx≈798 px /
  baseline≈4.24 mm; specimen_2 fx≈834 px / baseline≈3.97 mm. See
  `results/.../d4d_full_dataset/calibration_consistency.csv`.

## Quality filtering (transparent thresholds)
Reject if coverage <12 %, stereo/Zivid offset >60 ms, or implausible disparity. Warn if
offset >40 ms or tracker interpolation gap >500 ms. `quality_thresholds.json`.

## Coverage & synchronization
Valid GT coverage ~12–34 % of the endoscope frame (Zivid FOV is wider; only textured,
in-FOV, front-facing, z-buffer-winning surface). Stereo↔Zivid offset median ~14 ms.
Depth↔disparity round-trip exact (0.00 px).

## Splits (`splits/`, leakage-safe, deterministic)
Session-disjoint, specimen-disjoint, leave-one-specimen-out, and few-shot (1/2/4/8 sessions
× 3 seeds; 10/25/50 %). **Both anchors of a clip always stay together**; splits are at the
session or specimen level — never per-anchor.

## Permitted uses
Cross-dataset **sparse-keyframe** geometric accuracy evaluation; few-shot domain adaptation;
cross-backbone refinement; specimen/session-disjoint generalization.

## Limitations / warnings
- **Sparse, non-dense**: GT only at Zivid instants. **Do NOT** propagate a scan across a
  clip as GT — the scene is **non-rigid**.
- No dense per-frame temporal GT. Temporal use limited to a small validated neighborhood
  near an anchor (local pseudo-GT, not yet characterized).
- **Marker-identity assumption** (mire45_bridge): `polaris_spectra_MiRe45` ≡ `MiRe45`,
  corroborated by correct reprojection, not independently documented.
- Cross-camera photometric metric is an upper bound only.
- Some anchors lack the right rectified view (missing right stereo pair); left-referenced
  GT is unaffected.
- specimen_4 pending extraction; specimen_5 archive corrupt.

## Reproduce
```bash
python scripts/temporal_refinement/ood/d4d/d4d_keyframe_gt.py \
    --specimens specimen_1,specimen_2,specimen_3 --resume --workers 10
python scripts/temporal_refinement/ood/d4d/build_full_benchmark.py     # manifests + splits
python scripts/temporal_refinement/ood/d4d/validate_d4d_benchmark.py    # PASS / non-zero on fail
```
Canonical manifest: `manifests/benchmark_manifest.csv` (+ valid_only / valid_and_warning /
rejected). Transform chain version: `v2-multiconv`.
