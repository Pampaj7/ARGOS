# ARGOS v2 — Frozen Residual-Scale Audit

## Motivation

The large-scale causal BiDA audit establishes complementary t-1 evidence, while
the learned A2 residual refiner establishes a transferable correction direction.
The failure mode is not absence of geometric signal: full-strength A2 and
hard raw-versus-memory replacement can apply excessive updates, especially
under calibration or domain shift.  This audit isolates that question without
adding a network:

```text
d_lambda = d_raw + lambda * u_A2
```

and, for the already frozen balanced Raw Error authorization,

```text
d_lambda_auth = d_raw + lambda * a_raw_error * u_A2 .
```

`lambda` is a scalar, is never learned, and does not alter SEA-RAFT, the
canonical BiDA warp, A2, the raw-error detector, or any cache.  It is therefore
a controlled shrinkage/step-size experiment, not a new architecture.

The motivation is compatible with residual reconstruction, where a learned
correction is deliberately applied relative to an initial stereo estimate, and
with selective-prediction work in which risk and coverage must be evaluated
together.  It does **not** claim that a source-calibrated scale automatically
guarantees OOD safety; source/target risk can differ under covariate shift.

Relevant external context: [ResDepth](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Stucker_ResDepth_Learned_Residual_Stereo_Reconstruction_CVPRW_2020_paper.html),
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html), and
[Stereo Any Video](https://openaccess.thecvf.com/content/ICCV2025/html/Jing_Stereo_Any_Video_Temporally_Consistent_Stereo_Matching_ICCV_2025_paper.html).

## Immutable components

- frozen cached stereo disparity;
- frozen SEA-RAFT through `BiDAFlowInferenceAdapter`;
- causal t-1 alignment in `bidavideo.py`;
- frozen A2 checkpoint
  `results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt`;
- optionally frozen balanced raw-error detector and its serialized temperature
  and thresholds.

All artifacts are SHA256-verified by
`scripts/run_residual_scale_audit.py`.  No gradient graph is built.

## Selection protocol

The preregistered candidate scale grid is:

```text
0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00
```

Only `dataset_7_keyframe_1/2` and the three seen training backbones may select
a scale.  A candidate is eligible only if every seen backbone has positive EPE
gain, clean degradation at most 3%, and false-update rate at most 5% on this
calibration split.  The selected policy maximizes the minimum backbone gain;
ties favour the smaller scale.  A missing backbone row cannot pass the rule.

After serialization of `frozen_scale.json`, only the selected scale plus raw
and full-strength A2 controls are opened on `dataset_7_keyframe_3/4`.  Any
unseen-backbone or cross-domain evaluation is strictly subsequent and cannot
change the scale.

## Metrics and interpretation

All raw/refined comparisons use the existing paired cache-grid mask:

```text
GT coverage > 0.50 ∧ raw-valid ∧ aligned-valid ∧ warp-support.
```

Reports include EPE, Bad1/Bad3, boundary EPE, new-Bad3, coverage, intervention
precision, false-update rate, clean degradation, and update magnitude.  Frame
rows are aggregated pixel-weighted for geometry and separately per sequence;
no pixel count is used as an independent sample size.

Promotion requires more than positive EPE: the selected policy must retain a
nontrivial gain while meeting the predeclared safety limits on the final seen
split.  A scale that only reduces harm by making updates negligible is reported
as preservation, not as a geometry promotion.

## Results — SCARED-C-only, frozen A2

### Unconditional residual scale

On keyframes 1/2 the robust all-backbone selection chose `lambda=0.50`:
each backbone gained 0.0067--0.0077 px, with maximum clean degradation 1.97%
and maximum false-update rate 4.36%.  The final keyframe 3/4 result retained
positive EPE gain for RAFT-Stereo (+0.0129 px), S2M2-S (+0.0093 px), and
StereoAnywhere (+0.0122 px), but had roughly 6.0--6.4% clean degradation and
12.6--13.1% false updates.  This is **not** a safe configuration: reducing a
residual applied everywhere does not replace an applicability decision.

### Frozen Raw Error authorization times residual scale

The same source-only selection chose `lambda=1.00`, i.e. the already validated
raw-error-authorized A2 baseline.  A smaller global scale was never preferred
by the calibration objective.  On the final split all three backbones improved
geometry (+0.0125, +0.0157, and +0.0164 px for S2M2-S, RAFT-Stereo, and
StereoAnywhere), but false update remained 2.31--2.54% and clean degradation
1.42--1.64%.  Thus scalar shrinkage does not solve the central safety failure;
it only traces the expected gain--risk curve.

No unseen backbone or OOD dataset was opened by this audit.  The compact
artifacts are in `results/residual_scale_audit/` and
`results/authorized_residual_scale_audit/`; the latter is equivalent to the
existing frozen raw-error baseline at its selected scale, not a new promoted
model.
