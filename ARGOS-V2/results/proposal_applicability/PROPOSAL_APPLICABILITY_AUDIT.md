# ARGOS v2 — Proposal Applicability Audit

## Scientific question

This study tests whether the utility of the already frozen A2 proposal is more
predictable than raw disparity error.  It does not alter SEA-RAFT, BiDA, A2, a
stereo cache, or a disparity proposal.  The only trainable object is a small
proposal applicability detector which either accepts A2 exactly or returns the
raw cache value bit-exactly.

For a valid cache-grid pixel,

```text
e_raw = abs(d_raw - d_gt)
e_A2  = abs(d_A2  - d_gt)
u     = e_raw - e_A2
```

Positive utility means that A2 helps; negative utility means that it harms.  At
margin epsilon, labels are `helpful` for `u > epsilon`, `harmful` for
`u < -epsilon`, and `indifferent` otherwise.  Epsilon is selected only on
SCARED-C keyframes 1/2 from the predeclared ladder 0.05, 0.10, 0.25, 0.50 px.

## Frozen mechanism and exact sources

- A2: `model_design/models/learned_t1_refiner.py`, class
  `LearnedT1Refiner`, variant `A2`.  It consumes only disparity and validity
  evidence and returns
  `d_raw + g_error*c_memory*3*tanh(delta)`.  The validated checkpoint is
  `results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt`.
- BiDA: `model_design/external_components/bidavideo.py`; the flow/warp evidence
  is constructed by `scripts/run_learned_t1_refiner.py::build_evidence`.
- SEA-RAFT: loaded through `BiDAFlowInferenceAdapter`; evaluation mode,
  frozen parameters, and no gradient graph are retained.
- Existing proxy authorizer: `model_design/models/raw_error_detector.py`, class
  `RawErrorDetector`; checkpoint
  `results/raw_error_abstention/full/checkpoints/best_validation.pt` and frozen
  temperature/policy in `results/raw_error_abstention/full/operating_modes.json`.
- Exact abstention: `model_design/models/abstention.py::authorized_update` uses
  `torch.where`, so rejected pixels equal raw exactly and accepted pixels equal
  frozen A2 exactly.

The previous SCARED-C raw-error experiment reached 1.92% false updates and 0.97%
clean degradation, but retained only 63.2% of the full A2 EPE gain.  Its target
was whether raw error exceeded a threshold, not whether the proposed update was
beneficial.  That mismatch is the controlled variable in this study.

## Data, split, and mask contract

The dataset reuses `model_design/data/temporal_pair_dataset.py` and its masked
GT resize:

```text
resize(disparity * valid) / resize(valid)
```

with the disparity magnitude scaled to cache width 180.  The primary paired
mask is:

```text
gt_coverage > 0.50
and raw_valid
and aligned_validity
and warp_support
```

Every method is evaluated on this identical mask in positive-left cache-grid
pixels.  Sensitivity uses coverage 0.05, 0.25, 0.50, and 0.90.

The immutable group split is:

- train: the 13 accepted non-dataset-7 sequences;
- calibration/validation: `dataset_7_keyframe_1`,
  `dataset_7_keyframe_2`;
- final seen test: `dataset_7_keyframe_3`,
  `dataset_7_keyframe_4`;
- seen training backbones: S2M2-S, RAFT-Stereo, StereoAnywhere;
- Fast-FoundationStereo, CREStereo, and all OOD datasets remain inaccessible
  until one architecture, checkpoint, margin, and operating point are frozen
  and the seen promotion gates pass.

## Candidate tensor contract

All input maps are `[B,1,144,180]`, float tensors except explicit masks, and
contain no GT, backbone identity, or dataset identity.  The primary input has
23 channels:

1. normalized raw disparity;
2. normalized aligned t-1 disparity;
3. normalized A2 disparity;
4-5. signed and absolute A2 update;
6-7. signed and absolute raw/aligned disagreement;
8-10. A2 error gate, memory confidence, and pre-tanh delta;
11-13. raw-valid, aligned-valid, and warp-support masks;
14-15. raw disparity x/y gradients;
16-17. A2 disparity x/y gradients;
18-19. update x/y gradients;
20. flow magnitude;
21. photometric residual;
22. forward-backward error;
23. forward-backward confidence.

Normalization is fixed and backbone-independent: disparities `[0,64]/64`,
updates `[-3,3]/3`, disagreements `[-16,16]/16`, disparity gradients
`[-4,4]/4`, update gradients `[-3,3]/3`, flow `[0,32]/32`, FB error
`[0,8]/8`, and masks/confidences/photo `[0,1]`.

## Predeclared model and loss ladder

- P1: 1x1 pixel encoder and utility regression, approximately 1k-5k params
  (the supplied finite-difference maps make the end-to-end support 2 pixels).
- P2: three 3x3 local layers and utility regression, approximately 10k-50k
  params; learned receptive field 7x7 and end-to-end support 8x8 after the
  explicit forward-gradient evidence.
- P3: P2 plus positive Laplace uncertainty.
- P4: P3 plus helpful/indifferent/harmful logits and an explicit extra penalty
  for harmful pixels assigned helpful probability.

All candidate heads share the same evidence tensor.  Training may balance
helpful, indifferent, and harmful pixels, then raw-error/update magnitude and
boundary strata within those classes.  Validation/test retain the natural
pixel distribution.

P1/P2 use smooth-L1 utility loss.  P3 adds a stable heteroscedastic Laplace
term.  P4 adds three-class cross entropy plus the asymmetric harmful-as-helpful
penalty.  No unrelated loss or architecture is introduced.

## Authorization and model selection

The primary authorization is:

```text
u_hat > utility_margin
and sigma_u < uncertainty_threshold
and aligned_validity
and warp_support
and finite, bounded A2 update
```

P4 may additionally require the helpful class to be the most probable class.
Thresholds are selected on keyframes 1/2 only.  The output is
`where(authorized, d_A2, d_raw)`; there is no blending or new correction.

The selected model must, on keyframes 3/4, outperform the old Raw Error
Detector as a helpfulness proxy, reduce harmful acceptance, keep false updates
below 5% and clean degradation below 3%, retain at least 70% of the existing
balanced A2 gain, have nonzero coverage, and behave consistently on all three
seen backbones.  Only then are the unseen backbones loaded once.  OOD diagnostic
data are loaded only if that transfer stage passes.

## Target-audit requirements

Before model implementation, the frozen pipeline is used to report natural
utility distributions by backbone and sequence, all epsilon class fractions,
raw-error/utility correlation, raw-wrong-but-A2-harmful and
raw-clean-but-A2-helpful cases, and utility relations with update magnitude,
disagreement, flow, photometric and FB evidence.  These measurements determine
whether the target differs scientifically from raw-error detection; they do
not tune a model.
