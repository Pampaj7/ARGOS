# Strict causal t-1 utility-selector campaign

This directory is the final three-seed, **seen-domain** evaluation of the
lightweight causal selector over `{raw_t, BiDA-warped t-1}`.  It is a strict
protocol repair of the earlier dataset-7 reuse: complete sequence/session IDs
are disjoint across training, calibration, and final test.

## Frozen protocol

- Frozen stereo caches, SEA-RAFT, and canonical causal BiDA warp.
- Three seen training backbones: S2M2-S, RAFT-Stereo, StereoAnywhere.
- Train: dataset 1 keyframes 2--3, dataset 3 keyframes 1--4, dataset 6
  keyframes 1--4 (8,606 causal pairs per backbone).
- Calibration only: dataset 2 keyframes 2--4 (4,246 pairs per backbone).
- Final test only: dataset 7 keyframes 1--4 (4,052 pairs per backbone).
- Three independent training seeds: 0, 1, 2; 12 epochs each.
- Checkpoints are selected by validation constrained net utility, and the
  authorization threshold is calibrated only on dataset 2.

No Fast-FoundationStereo, CREStereo, or OOD dataset was loaded for fitting,
calibration, checkpoint selection, or evaluation: the seen promotion gate did
not pass.

## Final cache-grid result at coverage threshold 0.50

All values below are mean +/- sample standard deviation over training seeds;
geometry is pixel-weighted and confidence intervals in the seed artifacts are
sequence-unit bootstraps after collapsing repeated backbone measurements.

| Metric | Result |
| --- | ---: |
| Raw EPE | 0.533685 +/- 0.000000 px |
| Selected EPE | 0.521795 +/- 0.003057 px |
| EPE gain | 0.011890 +/- 0.003057 px |
| Raw-or-memory oracle gain | 0.052326 px |
| Oracle gain recovered | 22.72% +/- 5.84% |
| Authorization coverage | 0.966% +/- 1.072% |
| False-update rate | 0.597% +/- 0.852% |
| Clean-pixel degradation | 0.289% +/- 0.370% |
| Frames worsened | 8.998% +/- 5.485% |

Each seed has a positive EPE gain and passes the elementary safety gates, but
each recovers less than 50% of the available raw-or-memory oracle gain.  The
predeclared seen-domain promotion gate therefore fails.

**Decision: NO-GO for promoting this selector to unseen-backbone or OOD
evaluation.**  This does not negate the causal BiDA signal audit: aligned
t-1 memory has real oracle complementarity.  It establishes that the current
lightweight utility selector cannot retrieve enough of that signal at a stable,
paper-worthy operating point under strict session-disjoint validation.

## Artifacts

- `per_seed_summary.csv`, `aggregate_summary.json`: three-seed aggregation.
- `seed_*/split_audit.json`: exact frozen splits and pair counts.
- `seed_*/training_history.csv`, `calibration_metrics.csv`: training and
  calibration evidence.
- `seed_*/frame_metrics.csv`, `sequence_metrics.csv`, `backbone_metrics.csv`,
  `temporal_metrics.csv`: final D7 metrics.

The command used for each seed and the automatic calibration/evaluation chain
is preserved in `campaign.log` in the parent campaign launcher context and in
the per-seed `run.log`, `calibrate.log`, and `evaluate.log` files.
