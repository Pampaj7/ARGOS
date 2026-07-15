# ARGOS v2 — D0 Calibration Shift Audit

## Scope and immutable system

D0 is analysis only. It runs the frozen balanced composition recorded by
`results/ood_generalization/frozen_manifest.json`:

- frozen raw-error detector;
- frozen A2 bounded t-1 proposal;
- frozen SEA-RAFT / canonical BiDA evidence;
- frozen temperature and authorization thresholds.

It does not instantiate an optimizer, alter a parameter, fit a threshold, or
modify the detector, A2, or BiDA source. Source/checkpoint hashes are verified
before every run.

The design follows the ARGOS plan and SOTA state: a safe causal refiner must
earn the right to change already-good raw geometry, and motion-compensated
temporal agreement must not be mistaken for geometric correctness.

## Comparison protocol

`SCARED-C held-out` is the reference distribution: the three seen training
backbones on `dataset_7_keyframe_3/4`, exactly the final seen-test sequences.

| Dataset | Role | Geometry labels | Selection/tuning use |
|---|---|---|---|
| SCARED-C held-out | reference | internal processed cache-grid GT | reference only; frozen already |
| CREStereo | second unseen backbone | same SCARED-C GT | no tuning |
| SERV-CT | static / weak-sparse OOD | CT GT | no tuning |
| D4D | deformable OOD | final Zivid anchor only | no tuning |
| StereoMIS | in-vivo OOD diagnostic | none | no calibration claim |

StereoMIS uses a deterministic, evenly-spaced set of causal `t-1 -> t` pairs
per sequence. It is intentionally no-reference: it can describe detector and
feature shift, not AUROC, calibration, false updates, or geometry.

## Pixel definitions

Every evaluated frame is passed through the frozen pipeline. For all eligible
pixels the audit accumulates detector outputs, BiDA evidence, A2 update,
penultimate detector feature, and authorization. With GT:

```text
raw_error = abs(raw - GT)
raw_wrong = raw_error > 0.50 px
clean = raw_error <= 0.50 px
false_update = abs(update) > 0.05 px AND clean
clean_degradation = clean AND refined_error > raw_error + 0.02 px
incorrect_authorization = authorization AND
  (clean OR refined_error > raw_error + 0.02 px)
```

Classification/calibration metrics use the frozen cache-grid coverage 0.50,
raw-valid and BiDA-aligned-valid/common support. D4D applies those labels only
to its final anchor; other D4D transitions remain feature-shift observations.

## No-cache policy

Dense predictions, flows, detector maps, and feature maps are never written.
Univariate statistics, calibration counts and safety counts accumulate over
**every evaluated pixel**. A deterministic bounded pixel sample is retained in
memory only to estimate covariance, PCA, t-SNE, nearest-neighbor density,
Mahalanobis distance, feature overlap, and interpretable correlations. The
only persistent sample-level artifacts are low-dimensional PCA/t-SNE tables.

## Decision rules

D0 chooses among A–D only from frozen evidence:

- **A:** a shared pre-existing output threshold would separate harmful OOD
  authorization without excluding reference-domain helpful authorization;
- **B:** OOD failures are separated from SCARED-C in penultimate feature space
  with high support / density discrimination;
- **C:** the same feature regions overlap but calibrated error labels shift,
  indicating robust retraining is needed;
- **D:** harmful authorization is not predictable from the available universal
  evidence and penultimate representation.

No decision authorizes an OOD threshold change. The audit only identifies the
next experiment.
