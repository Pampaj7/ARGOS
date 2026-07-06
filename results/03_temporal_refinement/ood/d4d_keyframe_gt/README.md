# D4D sparse-keyframe stereo GT — final report

Turns D4D into a scientifically valid **sparse-keyframe** stereo benchmark from the Zivid
structured-light acquisitions. **No dense per-frame GT is fabricated** (non-rigid scene,
~2 Zivid scans/clip).

Pipeline: `scripts/temporal_refinement/ood/d4d/`. Processed GT:
`dataset/D4D/processed/keyframe_stereo_gt/`. Forensic audit:
`results/03_temporal_refinement/ood/d4d_keyframe_gt_audit/`.

## Result: 58 usable keyframe anchors (specimen_1)

| | |
|---|---|
| candidate clip-anchors | 86 (16 sessions, 43 clips × start/end) |
| converted successfully | 58 |
| failed (missing tf series / snr file) | 27 → honest reject |
| **benchmark-usable** | **58** (38 valid + 20 usable-with-warning) |
| valid GT coverage / anchor | 19–34 % (median 27 %) of the endoscope frame |
| depth↔disparity round-trip | **0.00 px (exact)** |
| stereo↔Zivid time offset | 1–55 ms (median 20 ms) |

## Final report (answers to the 10 questions)

1. **Depth `.npy` format/units**: float32, 2448×2048 (Zivid colour res), **metres**,
   optical-axis Z, NaN=invalid; backprojection with the Zivid colour K reproduces the `.ply`
   exactly. SNR is a pixel-aligned float32 map (0–~212).
2. **TF JSON format/direction**: `{timestamp, parent_frame, child_frame, transform:
   {translation xyz (m), rotation xyzw}}`; direction is **child→parent** (ROS), confirmed
   because the opposite put points behind the camera.
3. **Rectification**: OpenCV `initUndistortRectifyMap`/`remap` from `left/right(_rect).yaml`;
   fx = 798.32 px, baseline = 4.235 mm from the rectified P matrices.
4. **Zivid→left-camera chain** (the crux, validated — see
   `../d4d_keyframe_gt_audit/transform_chain_hypotheses.md`):
   `T_cam←zivid = inv(T_ps←cam)·T_ps←MiRe45·inv(T_polaris←MiRe45)·T_polaris←zivid`
   (endoscope tracked in `polaris_spectra`, Zivid in `polaris`, bridged by the MiRe45 marker;
   validated by anatomical/instrument reprojection alignment).
5. **Usable anchors**: 58 (38 valid, 20 usable-with-warning), 27 candidates rejected for
   missing transform/SNR data.
6. **Timestamp sync**: nearest stereo frame 1–55 ms (median 20 ms); pose interpolation gap a
   median 279 ms (flagged when large). Camera near-static over these intervals.
7. **Depth/disparity validity**: positive-px left-reference disparity 26–64 px; exact
   depth↔disparity round-trip; coverage ~27 %.
8. **Suitability**:
   - exact **sparse keyframe** stereo evaluation → **YES** (58 anchors).
   - **temporal consistency between anchors** → only as *local pseudo-GT* very near a scan
     (prepared, not run); the mid-clip interval is non-rigid and has no GT.
   - **local pseudo-GT near scans** → plausible for ±1–3 frames, needs the decay study.
   - **full dense temporal GT** → **NO** (would be fabricated).
9. **Remaining blockers**: marker-identity assumption (corroborated), cross-camera
   photometric metric is only an upper bound, session-wide epipolar table pending, specimen
   3–5 not extracted. See `../d4d_keyframe_gt_audit/blockers.md`.
10. **Recommended role in the ICRA paper**: use D4D as an **independent sparse-keyframe
    cross-dataset accuracy check** (real structured-light GT, different lab/scope than
    SCARED/SERV-CT) to corroborate per-frame accuracy claims. Do **not** use it for dense
    temporal-consistency numbers. It complements SERV-CT (dense but sparse-time) and the
    SCARED temporal metrics.

## Files
`benchmark_manifest.csv` (eval-ready), `anchor_quality.csv`, `rejected_anchors.csv`,
`quality_thresholds.json`, `quality_summary.json`, `methodology.md`, `keyframe_eval.csv`
(GT self-consistency sanity: 0 error), `environment_summary.txt`, `changed_files.txt`.

## Use
```bash
# regenerate GT (specimen_1):
python scripts/temporal_refinement/ood/d4d/d4d_keyframe_gt.py --specimen specimen_1
# quality + benchmark manifest:
python scripts/temporal_refinement/ood/d4d/finalize_d4d_benchmark.py
# evaluate any disparity predictions (per-anchor npy at 894x714):
python scripts/temporal_refinement/ood/d4d/evaluate_d4d_keyframes.py --pred-root <dir>
```
