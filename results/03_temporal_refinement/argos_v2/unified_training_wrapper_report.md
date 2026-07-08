# ARGOS v2 Unified Training Wrapper Report

Implemented a compact unified ladder wrapper for ARGOS v2 under `scripts/temporal_refinement/causal_bida/`.

## Files
- `configs.py`: resolved ladder configurations and shared training defaults.
- `train_argos_v2.py`: causal clip training, validation, checkpointing, resume, raw/eval-only support.
- `summarize_one_seed_ladder.py`: aggregate ladder comparison and decision gate.
- `launch_one_seed_ladder_p1i.sh`: H100 p1i launcher using GPU0/GPU1 concurrently.

## Common Semantics
- Seed: 0.
- Full-GT SCARED target shards with proposed balanced split.
- Certified target-to-source flow and warped previous validity through ARGOS v2 streaming helpers.
- Shared optimizer/loss schedule across trainable configs.
- Checkpoint selection: lowest validation refined MAE.
- Full final evaluation uses the certified streaming sequence evaluator path inside `train_argos_v2.py`.

## Smoke And Full Run
- Smoke tests passed for `current_only`, `aligned_local_faithful`, `faithful_causal_bida`, and `safe_causal_bida`; temporary smoke outputs were deleted.
- Full one-seed ladder completed on H100 LSF job `28884920`.
- Git commit: `b1b80a41d185775cb75e56ad3d707ed41e727790`.
