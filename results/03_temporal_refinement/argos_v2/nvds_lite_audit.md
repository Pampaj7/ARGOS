# ARGOS v2 NVDS-Lite Audit

Generated: 2026-07-07

## Executive Summary

NVDS-lite is not closed. There is no valid complete matrix and no valid complete minimal comparison that answers whether corrected explicit causal history genuinely helps.

The repository has the important correctness fixes in place:

- target-to-source warp flow is used by the cache builder;
- warped previous validity is used in the warp support mask;
- future-frame leakage tests pass;
- gradient graph validation passes;
- S2M2 and RAFT are not in the training graph;
- corrected gate/loss settings can depart from identity;
- the memory-heavy vectorised local-correlation implementation is not in current code.

The main remaining blocker is evaluation/protocol rather than flow/mask correctness: the current `eval_sequences` function resets history at non-overlapping 8-frame windows. That is acceptable for a rough diagnostic but not for a final causal streaming/sliding conclusion.

## Current Code State

Branch: `fable-wildcard-cfr`

Commit: `b1b80a41d185775cb75e56ad3d707ed41e727790`

Relevant files:

- `scripts/temporal_refinement/nvds_lite_causal/build_aux_cache.py`
- `scripts/temporal_refinement/nvds_lite_causal/model.py`
- `scripts/temporal_refinement/nvds_lite_causal/train_nvds_lite.py`
- `scripts/temporal_refinement/nvds_lite_causal/run_matrix.py`
- `scripts/temporal_refinement/nvds_lite_causal/validate_flow.py`
- `scripts/temporal_refinement/nvds_lite_causal/validate_causal_and_grad.py`
- `scripts/temporal_refinement/nvds_lite_causal/validate_identity_runtime.py`

The current model implements RGB-D/disparity encoding, low-memory local target-to-history correlation, causal history modes, bounded gated residuals, TGM, warp, safety and sparsity losses.

## Flow Convention Status

Corrected and validated.

- Cache builder stores `warp_flow[t-1] = RAFT(img[t] -> img[t-1])`.
- `warp_with_support` samples `source(p + flow(p))`.
- Therefore the stored flow pulls frame `t-1` into current frame `t`.

Validation over 24 SCARED frame pairs showed:

- backward/target-to-source photometric error: about `0.0107`;
- no-warp photometric error: about `0.0220`;
- forward/source-to-target photometric error: about `0.0303`.

One bookkeeping issue remains: `flow_mask_validation.json` contains a stale string field saying `cache_flow_stored` is forward. The actual builder and the validation decision both support target-to-source backward flow. This should be renamed later to prevent confusion.

## Mask Convention Status

Corrected in `clip_losses`.

The warp loss support mask uses:

- current validity;
- previous validity warped into current coordinates;
- in-bounds support;
- target-frame occlusion;
- border margin.

It does not intersect against unwarped previous validity.

## Identity Collapse Status

The old setup collapsed:

- gate bias `-2.0`;
- `lam_safe=0.5`;
- `lam_sparse=0.05`;
- modified-pixel ratio `0`.

The current setup is:

- gate bias `0.0`;
- `lam_safe=0.2`;
- `lam_sparse=0.02`.

A 300-step diagnostic showed nonzero correction:

- modified-pixel ratio about `0.645`;
- gate mean about `0.573`;
- residual abs p95 about `2.44` px;
- not saturated at the residual bound.

However, that diagnostic degraded validation geometry (`raw 4.9258 -> refined 5.0809`) and created high new-Bad3. This is a sign that the model is trainable but not yet scientifically positive.

## Current Experiment State

See `nvds_lite_run_manifest.csv` for row-level classification.

Counts from the manifest:

- `VALID_PARTIAL`: 4 rows (flow/mask validation, causal/grad validation, identity/runtime diagnostic, short timing run);
- `STALE_IDENTITY_COLLAPSE_SETUP`: 2 rows;
- `INCOMPLETE`: 3 rows;
- `FAILED_OTHER`: 2 rows;
- `VALID_COMPLETE`: 0 rows.

There are no complete valid NVDS-lite runs that can close the scientific question.

## Valid and Invalid Runs

Valid partial evidence:

- flow/mask validation;
- causality/gradient validation;
- identity/runtime diagnostic;
- short timing run.

Invalid/stale evidence:

- old smoke/validate-all logs with conservative identity-collapse setup;
- failed diagnostic pilot with output-path error;
- terminated matrix;
- matrix G0/G1 attempts without reusable artifact sets;
- prior local login-host launch attempts.

The D seed0 metrics printed in `logs/nvds_matrix_g1.log` are not enough to reuse because no matching checkpoint/config output directory was found.

## Known Bugs and Status

- Wrong flow direction: fixed in cache builder and validated.
- Unwarped previous validity: fixed in warp support.
- Future leakage: validation passes.
- Frozen S2M2/RAFT: validation passes; cache only.
- Identity collapse: fixed enough for trainability, but safety remains unproven.
- Vectorised local-correlation OOM: current matcher is low-memory.
- Artificial window resets: still present in current evaluator and must be addressed before final closure.

## Remaining Uncertainty

- Whether full-history NVDS-lite beats current-only under corrected settings.
- Whether full-history beats shuffled-history.
- Whether full-history beats naive concat.
- Whether warp loss helps.
- Whether RGB helps.
- Whether improvements, if any, are due to real causal correspondence or merely smoothing/overcorrection.

## Can NVDS-Lite Be Closed From Existing Runs?

No.

Existing evidence validates correctness pieces but not the core causal-history hypothesis. The current complete/stale logs cannot be combined scientifically.

## Minimal Reruns Needed

Run a one-seed closure only after fixing/adding proper streaming/sliding evaluation:

- raw S2M2 baseline;
- A: full RGB-D explicit local matching with warp loss;
- D: current-frame-only;
- E: shuffled-history;
- F: concat baseline;
- optionally B: full RGB-D without warp loss.

Do not run the old 18-run matrix unless the one-seed closure is positive and stable enough to justify promotion.
