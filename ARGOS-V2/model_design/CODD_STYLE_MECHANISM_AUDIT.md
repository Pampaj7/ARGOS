# ARGOS v2 CODD-style mechanism audit

## Purpose

The completed fixed-BiDA CODD-style Phase-1 result reports a 56.56% gain
normalised by the historical raw-versus-raw-τ-1 BiDA oracle.  That number is
not, by itself, an endpoint-selection recovery measurement: after the first
step of a four-pair causal clip, the model warps the preceding fused output,
and its output is a continuous convex interpolation.

This audit therefore distinguishes, on one common paired mask, the following
objects at current frame `t`:

* `d_S`: frozen current raw stereo disparity;
* `d_M_raw`: causal BiDA warp of frozen raw `d_S(t-1)`;
* `d_M_rec`: causal BiDA warp of the actual preceding fused state;
* `d_F`: the observed Phase-1 fused output.

The three reported ceilings are deliberately named differently:

* **historical-selection oracle**: `min(|d_S-d_GT|, |d_M_raw-d_GT|)`;
* **recurrent-selection oracle**: `min(|d_S-d_GT|, |d_M_rec-d_GT|)`;
* **convex-fusion oracle**: minimum error on the segment between `d_S` and
  `d_M_rec`, using the GT-only analytical coefficient
  `clip((d_GT-d_S)/(d_M_rec-d_S), 0, 1)`.

The corresponding normalised gains are never conflated.

## Fixed protocol

* Train IDs: 1, 3, 6; validation/checkpoint selection: ID 2; final test: ID 7.
* Three frozen Phase-1 checkpoints are evaluated without retraining for the
  post-hoc part.
* The common paired evaluation support is the intersection of GT coverage
  above 0.50, raw validity, historical-memory validity/support, and
  recurrent-memory validity/support.  Its equality with the original Phase-1
  historical mask is measured rather than assumed.
* SEA-RAFT, BiDA, stereo caches, the frozen ResNet context extractor, and the
  selected Phase-1 checkpoint are no-gradient artifacts.
* A causal four-frame clip starts from raw `t-1` and resets only at its clip
  boundary.  Continuous streaming is a separate no-future evaluation which
  resets only at a true sequence/backbone boundary.

## Confirmatory ablations

After the post-hoc report is complete, the only allowed configurations are:

1. the existing full recurrent soft-fusion reference;
2. raw-previous-memory (no recurrence), all other Phase-1 details fixed;
3. no learned stereo evidence (remove frozen-feature L/R costs, frozen-feature
   appearance correlations, their support maps, and frozen feature context);
4. hard endpoint selection derived from the complete frozen model, with its
   threshold chosen on ID 2 only and bit-exact raw fallback.

No SE3/RAFT3D motion, new cue family, backbone, OOD dataset, or capacity
change belongs to this audit.

## Interpretation guardrails

* A positive `e_best_endpoint - e_F` is interpolation advantage: it cannot be
  described as correct endpoint selection.
* `e_M_raw - e_M_rec` quantifies temporal-candidate improvement caused by
  recurrent state, not fusion quality by itself.
* Continuous-streaming results are not substituted for clip-reset results.
* The full soft output does not have bit-exact abstention because sigmoid
  weights are generally nonzero.  Hard-output safety is a separately reported
  diagnostic.
