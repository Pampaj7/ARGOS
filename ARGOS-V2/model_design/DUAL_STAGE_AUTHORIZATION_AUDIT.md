# ARGOS v2 — Dual-Stage Authorization Audit

## Question and frozen composition

This experiment asks whether the validated Raw Error Detector can retain A2
recall while the frozen P4 proposal-applicability detector is used only as a
veto.  It introduces no proposal, learned feature, loss, or training.

```text
a_final = a_raw_error AND NOT a_veto
d_out   = where(a_final, d_A2, d_raw)
```

P4 can close an existing authorization but can never open one.  Rejection is
bit-exact raw and acceptance is bit-exact frozen A2.

The exact frozen sources are:

- `model_design/models/learned_t1_refiner.py::LearnedT1Refiner`, A2 checkpoint
  `results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt`;
- `model_design/models/raw_error_detector.py::RawErrorDetector` and
  `model_design/models/abstention.py::authorization_mask`, checkpoint and
  balanced temperature/thresholds under `results/raw_error_abstention/full/`;
- `model_design/models/proposal_applicability_detector.py`, P4 checkpoint and
  normalization under `results/proposal_applicability/P4/`;
- `model_design/external_components/bidavideo.py` and frozen SEA-RAFT through
  `BiDAFlowInferenceAdapter`.

No gradient graph is constructed for these components.

## Why composition is justified

On final seen SCARED-C, the Raw Error authorization retains 0.026946 px gain
with 75.63% intervention precision, 1.90% false updates, and 0.98% clean
degradation.  Standalone P4 raises precision to 84.11% and lowers false update
and clean degradation to 1.19% and 0.40%, but retains only 49.98% of the Raw
Error gain.  The controlled hypothesis is therefore that P4's harmfulness
signal is useful as a conditional veto rather than as a replacement gate.

## Split, mask, and leakage policy

- Policy audit/selection: `dataset_7_keyframe_1` and
  `dataset_7_keyframe_2` only, for S2M2-S, RAFT-Stereo, StereoAnywhere.
- Final seen: `dataset_7_keyframe_3` and `dataset_7_keyframe_4`, opened only
  after the selected policy and hashes are frozen.
- Fast-FoundationStereo and CREStereo are inaccessible until final seen passes.
- OOD datasets are inaccessible until unseen-backbone transfer passes.
- Primary common mask: GT coverage >0.50, raw valid, aligned valid, and warp
  support.  Sensitivity is 0.05/0.25/0.50/0.90.
- All disparity and threshold values are cache-grid pixels at width 180.

The conditional audit population is the set of nontrivial proposals
(`abs(update)>0.05`) already authorized by the frozen balanced Raw Error
Detector.  At epsilon 0.10, utility >epsilon is helpful, utility <-epsilon is
harmful, and the remainder is indifferent.  Historical safety/precision uses
the established 0.02-px material-change margin so it is directly comparable
with the validated baseline.

## Predeclared minimal ladder

### C0 — frozen Raw Error authorization

No veto.  This must reproduce the validated authorization exactly.

### C1 — update-magnitude veto

Reject an existing authorization when pixel update magnitude exceeds one of:

```text
0.10, 0.25, 0.50, 1.00, 2.00, 3.00 px
```

A 5x5 local-mean magnitude uses the same thresholds and is retained only if the
pixel rule is unstable across validation sequences.  This direction is a
safety veto, not the previously validated standalone lower-bound selector.

### C2 — frozen P4 veto

Evaluate compact one-signal rules:

- harmful probability >= 0.10, 0.25, 0.50, 0.75, 0.90;
- predicted utility <= -0.25, -0.10, 0.00, 0.05 px;
- uncertainty >= 0.05, 0.10, 0.25, 0.50, 1.00 px;
- P4 argmax class is harmful.

Also evaluate the compact conjunction
`harmful_probability >= p AND predicted_utility <= u` for
`p in {0.25,0.50,0.75}` and `u in {-0.10,0.00,0.05}`.  No larger sweep is
allowed.

### C3 — combined veto

Combine selected C1 and C2 by logical OR only if C1 correctly rejects at least
5% of the baseline harmful proposals that selected C2 misses, without rejecting
more helpful than harmful proposals in that unique subset.  Otherwise C3 is
scientifically unjustified and remains unpromoted.

C4 is not run unless every binary C2 rule loses excessive gain despite a
clearly separable conditional harmfulness distribution.

## Conditional metrics and selection

Every candidate reports harmful recall/precision, useful-proposal retention,
conditional harmful acceptance, conditional helpful rejection, veto rate,
gain retained versus C0, and recovery toward the oracle conditional veto.

Balanced selection on keyframes 1/2 maximizes EPE gain among policies satisfying:

- gain retained >=80%;
- false updates <1.25%;
- clean degradation <0.60%;
- intervention precision >80%;
- intervention coverage >=0.5%.

If no policy meets all constraints, the highest-gain nondominated safety policy
is frozen but is explicitly ineligible for promotion.  A safety operating point
may additionally target false updates <1% and clean degradation <0.5%.

The final promotion gate repeats those constraints on keyframes 3/4, requires
all three seen backbones to improve and no catastrophic sequence degradation,
then and only then allows one frozen unseen-backbone evaluation.

## Required baselines

Final reports include raw, unconditional A2, C0, prior standalone P4,
standalone high-update selection, selected C1, selected C2, justified C3,
oracle conditional veto, and deterministic random veto matched to the selected
conditional rejection rate.  All methods share the identical paired mask.

## Interpretation

A positive result requires the veto to remove conditional harmful proposals,
not simply remove most proposals.  The main evidence is retained C0 gain at
improved safety and precision.  Failure to retain 70% is a NO-GO; 70-80% is a
partial GO; at least 80% with all safety constraints is a GO.

