# ARGOS v2 Aligned-Local-Only Readiness Audit

## Verdict

The repository is mostly ready to implement aligned-local-only after NVDS-lite is closed. The most reusable pieces are flow/mask/cache/evaluation infrastructure. The main missing pieces are the actual aligned-local-only model and a final streaming evaluator.

## Reusable Infrastructure

- Full-GT S2M2 target shards: `results/03_temporal_refinement/evaluation/s2m2_gt_refiner_targets_full`.
- RGB/flow/occlusion cache: `results/03_temporal_refinement/nvds_lite_causal_pilot/aux_cache`.
- Correct target-to-source flow convention in `build_aux_cache.py`.
- Backward grid-sampling warp in `train_nvds_lite.py::warp_with_support`.
- Warped previous validity and occlusion masks in `clip_losses`.
- Sequence-disjoint split from `refiner_failure_analysis/proposed_balanced_split.json`.
- Safety/sparsity/spatial/TGM/warp loss code.
- Causality and gradient validation scripts.
- Low-memory local feature matching pattern.
- H100 runtime diagnostics.

## Missing Implementation Blocks

Aligned-local-only should add:

- current raw disparity/RGB input;
- previous raw disparity warped into current coordinates;
- previous RGB warped into current coordinates;
- signed disparity difference;
- absolute disparity difference;
- reliability/support mask;
- optional disparity gradients/local statistics;
- compact convolutional local encoder;
- bounded gated residual output.

No persistent hidden state yet. No future frame. No BiDA propagation yet.

## Potential Risks

- Evaluator currently uses non-overlapping window resets; fix before comparing aligned-local-only against NVDS-lite.
- Need avoid reintroducing source-to-target flow confusion.
- Need ensure flow scaling remains correct at the target resolution.
- Need ensure low-valid/occluded regions do not create false correction support.
- Need avoid raw-good false activation, especially for SERV-CT-style low-error regimes.

## Proposed File Locations

- `scripts/temporal_refinement/argos_v2_aligned_local/model.py`
- `scripts/temporal_refinement/argos_v2_aligned_local/train.py`
- `scripts/temporal_refinement/argos_v2_aligned_local/evaluate.py`
- `scripts/temporal_refinement/argos_v2_aligned_local/validate.py`

Alternatively, keep under `scripts/temporal_refinement/nvds_lite_causal/` only if the code remains a baseline family. A separate `argos_v2_aligned_local/` namespace is cleaner.

## Minimal Smoke-Test Design

1. Load one sequence from the existing shards/cache.
2. Warp previous RGB/disparity/validity into current coordinates.
3. Confirm no NaNs/Infs and support mask coverage is plausible.
4. Run model forward on a short clip.
5. Verify identity-safe init.
6. Perturb future frames and confirm outputs at earlier frames are unchanged.
7. Backprop one mixed loss and verify all trainable parameters get finite gradients.
8. Run 50-100 training steps and confirm nonzero bounded corrections without saturation.

## No Implementation

This audit does not implement aligned-local-only. It only confirms readiness and lists the smallest missing blocks.
