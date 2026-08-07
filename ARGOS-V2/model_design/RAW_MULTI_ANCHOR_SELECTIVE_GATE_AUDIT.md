# ARGOS v2 — Raw Multi-Anchor Selective Gate Audit

## Scientific scope

The frozen raw multi-anchor adapter retrieves one immutable raw anchor from ages
`{1, 2, 4, 8}` and proposes a soft correction of the current raw disparity.  Its
dataset-7 geometry is validated (raw EPE 0.547926, output EPE 0.510474), but its
authorization is unsafe (35.06% harmful accepted interventions, 4.86% clean-pixel
degradation).  This study changes neither retrieval nor fusion.  It adds only a
post-hoc reject option that predicts whether that exact frozen proposal will harm
the current raw disparity.

The operational contract is deliberately **veto-only**.  The frozen policy first
constructs its selected candidate, fusion weight, proposed output, and existing
eligibility mask.  The new gate may close an eligible intervention; it may never
open one rejected by the frozen policy, alter the selected age, change the fusion
weight, or write its output into the raw anchor bank.  Rejection returns the raw
tensor bit-exactly.

## Targeted SOTA extraction

The following conclusions were extracted before implementation from
`SOTA/codd.pdf` and `SOTA/stereo-temporal-canidate-selection.md`.

- CODD is the closest temporal-fusion baseline.  It compares current stereo with
  aligned temporal disparity and supervises per-pixel reset/fusion weights from
  their relative GT errors using dead bands.  CODD therefore supports the use of
  intervention-relative evidence, but it performs continuous convex fusion and
  does not formulate an explicit calibrated reject policy or report selective
  risk/coverage and harmful-update rates.
- Learning-to-defer and reject-option theory identifies the predicted difference
  between the two actions' errors as the appropriate decision quantity.  A
  confidence-only cascade can be suboptimal when the alternative is a specialist,
  which accurately describes an old aligned disparity: very useful in some regions
  and harmful in others.
- SelectiveNet and classical reject-option work motivate a default action and a
  risk/coverage curve, not threshold-free discrimination alone.  Here the safe
  default is the current raw disparity and coverage is accepted temporal updates.
- Stereo confidence and evidential stereo uncertainty estimate marginal reliability
  of a disparity.  They are adjacent but not equivalent to the counterfactual
  target `|d_raw-d_gt| - |d_proposed-d_gt|`; a proposal can be harmful even where
  the raw map itself is uncertain.
- Decision-curve and failure-detection literature warns that useful AUROC can
  coexist with negative net benefit at the deployed operating point.  Selection
  must therefore be chosen by realized validation utility under explicit safety
  constraints, with risk/coverage and the never-intervene baseline reported.
- Learn-then-Test, conformal risk control, and conformal decision methods offer
  calibration machinery.  Dense pixels and nearby crops are correlated, however.
  This study makes no pixel-level conformal guarantee; any uncertainty bound uses
  frames or sequence/frame groups as calibration units and states the small
  effective sample size.

## Frozen artifacts and proposal contract

- Frozen checkpoint:
  `results/raw_multi_anchor_temporal_refiner/soft_fusion/checkpoints/best_validation.pt`
- Expected SHA-256:
  `40526a32ef6e9a62a3ea2b59e6751a60c441b8190f9b96522e3b12b35895d5cd`
- Immutable candidates: direct current-to-anchor BiDA warps of raw disparities at
  ages 1, 2, 4, and 8.
- Frozen proposal:
  `d_proposed = (1-w) * d_raw + w * d_anchor`.
- Gate output:
  `where(frozen_eligible & gate_accept, d_proposed, d_raw)`.
- Stereo backbones, SEA-RAFT, BiDA, proposal features, scorer, retrieval, fusion,
  anchor manifests, and dataset splits remain frozen and gradient-free.

## Counterfactual labels

Training-only labels use the common valid GT support:

```
delta  = abs(d_raw - d_gt) - abs(d_proposed - d_gt)
y_harm = delta < -harm_margin
y_help = delta >  help_margin
```

No raw/proposed error, GT-derived label, dataset identity, or backbone identity is
an inference feature.  Margins are chosen on dataset 2 from a predeclared compact
set containing zero and small practical cache-pixel margins.

## Compact inference features

The gate reuses only causal proposal evidence: frozen top-1 score and top-1/top-2
margin, fusion weight, selected age, signed and absolute anchor/raw disagreement,
actual update magnitude, temporal median and MAD, raw and selected-anchor deviation
from the median, valid-anchor count, anchor agreement count, selected support and
forward/backward confidence, selected validity, and local mean/variance of
disagreement and update magnitude.  A simple aligned-image photometric residual is
permitted only as a separately identified compact feature.  There is no learned
appearance encoder.

## Predeclared comparison and selection

The comparison ladder is: frozen ungated reference, frozen-score threshold,
update-magnitude threshold, logistic harm gate, compact harm-only MLP, and compact
joint gain/harm MLP.  Monotonic risk-weighted shrinkage is optional and follows the
binary reject experiment.  Neural gates remain below 100k parameters.

Checkpoint, margins, and operating thresholds use dataset 2 only.  The selected
policy maximizes validation EPE gain subject to accepted-harm <=10%, clean
degradation <=3%, and degraded frames <=25%; complete curves and 5/10/15% accepted
harm operating points are retained.  Dataset 7 stays inaccessible until a frozen
manifest records the architecture, checkpoint, margins, thresholds, and hashes.

## Split and leakage contract

- Train: dataset IDs 1, 3, 6.
- Validation/calibration: dataset ID 2.
- Frozen test: dataset ID 7.
- Sequence/frame identifiers remain attached to sampled proposal rows.
- Dataset 7 is held out from this experiment but is not a pristine project-wide
  holdout because prior ARGOS v2 studies inspected it.
- No future frame, GT inference feature, candidate reopening, candidate mutation,
  output-to-bank write, or cross-sequence state is allowed.

## Decision rule and claims

The joint policy accepts only when the frozen proposal is eligible,
`delta_hat > tau_gain`, `p_harm < tau_harm`, and the selected candidate is valid.
Otherwise it returns raw bit-exactly.  A full GO additionally requires beating the
fixed H=4 reference on dataset 7, accepted harm <=10%, clean degradation <=3%,
degraded frames <=25%, nontrivial coverage, and positive gain across all seen
backbones and nearly all sequences.

This experiment may establish safer selective intervention on held-out SCARED-C
sequences and compatibility with three training-seen stereo backbones.  It cannot
establish clinical safety, formal conformal coverage, unseen-backbone transfer, or
external-domain robustness.
