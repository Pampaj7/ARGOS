# ARGOS v2 — dataset-robust causal utility-selector audit

## Why this audit is necessary

The causal BiDA signal is established independently of any learned policy:
the large-scale audit has a positive raw-versus-aligned-t-1 oracle gain in all
five accepted SCARED-C dataset/session IDs and on the two unseen backbones.
However, the original utility-selector protocol uses `dataset_7` keyframe
groups for both validation and final test.  Those groups share a higher-level
acquisition/session ID, so they cannot substantiate an acquisition-general
paper claim.

The completed deterministic L1/ZNCC, census, and left-right consistency
audits do not provide a safe candidate-quality cue.  Adding another local
confidence map is therefore not justified.  The smallest remaining
evidence-based intervention is to remove an avoidable *training-distribution*
imbalance and evaluate at the actual independent unit: dataset/session ID.

## Frozen parts

This study does not change the selector CNN, its 13 universal causal inputs,
the utility labels, GT resizing, cached disparities, SEA-RAFT, BiDA warp,
or any metric.  In particular, the network receives neither dataset nor
backbone identity.  It has no future frames and no recurrent state.

## Hierarchical sampling configuration

The legacy sampler makes every `(backbone, sequence)` equally frequent.  It
therefore gives a dataset ID with four accepted keyframe sequences twice the
exposure of one with two sequences.  The new sampler is an input-order-only
configuration:

1. equalise `(backbone, dataset-ID)` groups;
2. within each group, equalise complete keyframe sequences;
3. within a sequence, shuffle causal pairs deterministically and repeat only
   to equalise exposure.

Every original causal pair is seen at least once in each epoch.  Extra draws
only balance under-represented groups; no prediction/flow cache is created.
This is a limited form of pre-specified group-robust sampling, not a learned
domain classifier or a group label at inference.  It is motivated by the
standard group-shift objective of controlling performance on pre-specified
groups, while retaining strong regularisation because naive group-DRO can
overfit ([Sagawa et al., ICLR 2020](https://openreview.net/pdf?id=ryxGuJrFvS)).

## Nested leave-one-dataset-ID-out protocol

The accepted session IDs and sequence counts are:

| dataset ID | accepted sequences |
|---|---:|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 6 | 4 |
| 7 | 4 |

For every fold, all keyframes of the test dataset ID are unseen during
training and calibration.  A distinct dataset ID is used for calibration;
the remaining IDs are training only.  Sequence, validation, and test sets are
checked for both sequence and dataset-ID disjointness before a run starts.
Calibration thresholds are selected solely on the validation session.

The first pilot is `test=7`, `validation=2`, `train={1,3,6}`.  It directly
tests whether the valid BiDA signal transfers from three sessions to a fourth,
without the historical dataset-7 leakage.  It must show a feasible safe
policy before any complete five-fold/three-seed campaign is justified.

Checkpoint choice is also frozen to validation: among epochs whose
quantile-derived policy has at least `.2%` coverage and at most `.25`
harmful-to-helpful utility cost, the selected checkpoint maximises net
validation utility.  Training loss is reported for convergence only; it is
not the deployment selection criterion.  The already-running first seed was
started before this clarification and is therefore labelled a convergence
pilot rather than a final seed if its loss-selected checkpoint differs from
the utility-selected one.

The frozen legacy-policy calibration grid is probability
`{.50,.60,.70,.80,.90}`, expected utility in cache pixels
`{0,.005,.01,.02,.05,.10}`, and predicted harmful magnitude
`{.025,.05,.10,.25,.50}`.  It is evaluated only on the designated calibration
session; the final-test and unseen backbones never enter this grid.

## Gates

The primary unit is a complete dataset/session, with its three seen-backbone
measurements averaged before uncertainty estimation.  A configuration is not
promoted merely for a pixel-weighted gain.  The pilot must produce a positive
test-session EPE gain at a non-zero coverage and meet the existing safety
limits (false update `<2%`, clean degradation `<1%`).  A complete result must
then use nested folds, sequence/session bootstrap, and frozen unseen-backbone
evaluation only after the seen-session gate passes.

The result will distinguish two hypotheses:

* **sampling-limited**: the same causal evidence is usable after balanced
  cross-session training;
* **information-limited**: even a session-disjoint, balanced configuration
  cannot select it safely, so further confidence-only variations should stop.
