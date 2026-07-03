# Distillation Planning

Source run: `results/03_temporal_refinement/evaluation/gt_temporal_rectified_streaming_s2m2_v2_artifact_temporal`

This folder is planning only. No model training, RAFT, StereoAnyVideo, DINO extraction, or prediction cache generation was run.

## Why These Outputs Exist

The full no-cache S2M2-S artifact-temporal evaluation shows broad geometry degradation plus concentrated stress cases. The next primary direction is lightweight teacher/oracle distillation: use selected clean and failure clips to define residual, confidence, artifact-mask, and blending targets for a cheap online refiner.

DINO is secondary for now: it should be a feature-ablation/probe to test whether frozen tissue/appearance features predict S2M2 high-error, boundary artifact, flicker, or mismatch regions. It should not become a full feature-cache pipeline unless the probe is useful.

## Files

- `sequence_failure_taxonomy.csv`: per-sequence ranks, thresholds-derived group, and failure reasons.
- `candidate_clips_for_distillation.csv`: compact 50-150 frame contiguous clips for future teacher/oracle target generation.
- `candidate_frames_for_dino_probe.csv`: balanced frame subset for a DINO probe, not extracted features.
- `distillation_targets_plan.json`: target definitions and future file expectations.
- `dino_probe_plan.json`: frozen-DINO probe design, splits, metrics, and storage policy.

## Sequence Groups

Groups are quantile-based from existing CSV metrics: `clean_core`, `high_boundary_error`, `high_temporal_flicker`, `high_motion_mismatch`, `low_valid_stress`, and `catastrophic_geometry`. Catastrophic geometry uses the top tail of disparity MAE, bad-3px, and depth MAE composite. Low-valid and artifact groups use q25/q75 stress thresholds.

## Next Step

Generate teacher/oracle targets only for `candidate_clips_for_distillation.csv`. Keep DINO as a probe over `candidate_frames_for_dino_probe.csv`; save compressed patch/token features only if the first probe needs persistence.
