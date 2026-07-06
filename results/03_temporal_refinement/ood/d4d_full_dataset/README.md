# D4D full-dataset keyframe benchmark — final report

Scales the validated D4D sparse-keyframe stereo-GT pipeline to all available specimens.
**No dense per-frame GT fabricated.** No raw data modified.

Code: `scripts/temporal_refinement/ood/d4d/`. Processed GT + manifests + splits:
`dataset/D4D/processed/keyframe_stereo_gt/` (+ `DATASET_CARD.md`). Per-anchor validation +
transform-chain proof: `../d4d_keyframe_gt_audit/`.

## Final summary (12 points)

1. **Specimens processed**: 3 (specimen_1, 2, 3). specimen_4 extraction in progress
   (137 GB, resumable); specimen_5 archive corrupt (0.3 MB, not gzip).
2. **Sessions / clips**: 40 sessions; ~100 clips (start+end anchors).
3. **Total candidate anchors** converted: 197 (+5 conversion-rejected: missing tf/geometry).
4. **Valid / warning / rejected**: 142 valid, 24 usable_with_warning, 31 rejected →
   **166 usable**.
5. **Per specimen**: specimen_1 82 (60 valid), specimen_2 39 (31 valid), specimen_3 76
   (51 valid). See `specimen_inventory.csv`.
6. **Main rejection reasons**: low valid coverage (<12 %), high stereo/Zivid offset
   (>60 ms), and (at conversion) missing tf series. `global_rejected_anchors.csv`.
7. **Coverage distribution**: 12–34 % of the endoscope frame (median ~22 %). `benchmark_summary.json`.
8. **Synchronization**: stereo↔Zivid offset 0–55 ms (median ~14 ms); pose interpolation
   flagged when >500 ms.
9. **Calibration consistency**: calibration is **per-specimen** (specimen_1 fx≈798/base≈4.24 mm;
   specimen_2 fx≈834/base≈3.97 mm; specimen_3 direct_ps). Recorded, not forced —
   `calibration_consistency.csv`. **Three tf conventions** auto-detected and each validated
   by reprojection: `mire45_bridge`, `direct_ps`, `direct_polaris`.
10. **Split sizes**: session-disjoint (70/15/15), specimen-disjoint, leave-one-specimen-out
    (3 folds), few-shot (1/2/4/8 sessions × 3 seeds; 10/25/50 %). All leakage-safe
    (clip-atomic, session/specimen-level). `split_summary.json`, `splits/`.
11. **Remaining blockers**: specimen_4 pending extraction (pipeline resumable — one command
    folds it in); specimen_5 corrupt; marker-identity assumption for mire45_bridge
    (corroborated); some anchors missing the right rectified view (left GT unaffected);
    dense temporal GT impossible (non-rigid). `../d4d_keyframe_gt_audit/blockers.md`.
12. **Recommended ICRA paper usage**: an **independent, real-structured-light sparse-keyframe
    cross-dataset accuracy check** (multiple specimens, different scope/lab than
    SCARED/SERV-CT), plus specimen-disjoint few-shot adaptation studies. **Not** for dense
    temporal-consistency numbers — use SCARED temporal metrics for that.

## Pipeline (resumable, idempotent, parallel)
`d4d_keyframe_gt.py` — dual-layout session discovery, per-session tf caching, convention
auto-detection, z-buffered projection; options `--specimens --resume --workers --dry-run
--no-diag`. `build_full_benchmark.py` — inventory + quality + canonical manifest + splits.
`validate_d4d_benchmark.py` — 47 checks, **PASS** (exit non-zero on failure).
`evaluate_d4d_keyframes.py` — accepts arbitrary disparity predictions.

**Canonical command** (deterministic):
```bash
python scripts/temporal_refinement/ood/d4d/d4d_keyframe_gt.py \
    --specimens specimen_1,specimen_2,specimen_3 --resume --workers 10
python scripts/temporal_refinement/ood/d4d/build_full_benchmark.py
python scripts/temporal_refinement/ood/d4d/validate_d4d_benchmark.py
```
When specimen_4 finishes extracting, add `specimen_4` to `--specimens` and rerun (already-done
anchors are skipped).

## Report files
`extraction_inventory.csv`, `extraction_summary.json`, `disk_usage_summary.json`,
`specimen_inventory.csv`, `session_inventory.csv`, `calibration_consistency.csv`,
`global_anchor_quality.csv`, `global_rejected_anchors.csv`, `benchmark_summary.json`,
`validation_report.{json,md}`, `split_summary.json`, `environment_summary.txt`,
`changed_files.txt`, `diagnostics/`.
