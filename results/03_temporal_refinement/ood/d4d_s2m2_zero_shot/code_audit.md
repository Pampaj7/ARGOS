# Code & checkpoint audit — D4D zero-shot

Reuses the already-validated ARGOS inference paths (no second S2M2 implementation).

## S2M2-S backbone
- Build: `S2M2(feature_channels=128, dim_expansion=1, num_transformer=1, use_positivity=True,
  refine_iter=3)` — `scripts/temporal_refinement/data_prep/predict_s2m2_long_sequences.py`.
- Checkpoint: `external/frame_stereo_repos/s2m2/weights/pretrain_weights/CH128NTR1.pth` (pretrained, non-surgical).
- Recipe (`infer`): resize to width 512 (isotropic), pad /32, fp16 autocast, crop, upsample
  disparity back to native and divide by scale → positive-px disparity in ORIGINAL image coords.
  Identical to the SCARED `s2m2_s512` targets the refiners were trained on.

## Refiners (SCARED-trained, zero-shot)
- Registry: `scripts/temporal_refinement/ood/eval/model_registry.py` — pins each model's
  selected primary checkpoint, residual scale, and application policy. See `checkpoint_inventory.csv`.
- Input contract: 16-channel feature stack (`make_features_from_raws`) = 4 causal raw-disp
  frames + 4 valid masks + {dt1, mean, median, var, |raw-median|, gx, gy, edge}, all ÷DISP_SCALE=64,
  at target_scale=0.25 grid. Context = [t, t-1, t-2, t-3] causal (no future).
- Application: internal_gate (refined=raw+residual) for EGBM v1/v2/v2-CARE/v3-CARE-S window,
  MPC, CPV; threshold_gate (refined=raw+(p_bad>=0.7)·residual) for v3.2c. No D4D tuning.
- MPC/CPV are large-correction analytical branches (residual scale 32) — reported but NOT the
  main deployable comparison (per task).

## D4D evaluator
- Reuses `evaluate_ood_refiners.frame_metrics` (accuracy + correction-safety battery) and the
  D4D keyframe GT (`dataset/D4D/processed/keyframe_stereo_gt/manifests/`).

## Resolved (not guessed)
- Disparity sign/units, resize/scale, DISP_SCALE, context length, causal order — all taken from
  the validated SCARED/SERV-CT paths, unchanged.
