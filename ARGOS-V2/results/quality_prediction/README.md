# ARGOS v2 — Q0 quality prediction

## Decision

**NO-GO for promoting this Q0 predictor to Q1 selection.** The model learns a
useful image/candidate difficulty and uncertainty signal, but it does not learn
the relative quality ordering required to identify the minority correct memory.
No Q1 selector or Q2 refiner was implemented.

## Protocol

- Candidates: raw, BiDA-aligned t-1, t-2, t-4, t-8.
- Grid: cache grid 144x180 only; disparity units are pixels at width 180.
- Primary GT coverage: 0.50; error sensitivity also evaluated at 0.05, 0.25,
  and 0.90.
- Training backbones: S2M2-S, RAFT-Stereo, StereoAnywhere, balanced.
- Training sequences: all 13 accepted non-dataset-7 sequences, with a
  deterministic contiguous cap of 128 records per sequence/backbone (4,992
  temporal records).
- Held-out: dataset_7_keyframe_1 through dataset_7_keyframe_4, 300 records per
  sequence/backbone (3,600 records).
- Fast-FoundationStereo and CREStereo were rejected by the dataset/runner and
  were not loaded, tuned, diagnosed or evaluated.
- RGB was used only by frozen SEA-RAFT/BiDA. The Q0 predictor receives no RGB
  and no backbone identity.

## Selected model

The pilot selected a shared pixel-wise encoder with trainable Laplace
uncertainty heads: 19 universal evidence channels, two shared 1x1 convolution
layers, shared positive `mu` and `sigma` heads, and a diagnostic advantage
head. Candidate age is an input; encoder parameters are shared across raw and
all four memory ages. Total trainable parameters: **1,155**.

The larger local/Mini-U-Net models did not win the capacity-controlled pilot.
The key pilot results were:

| Variant | Spearman | Diagnostic regret | Uncertainty/error Spearman |
|---|---:|---:|---:|
| Q0-1 absolute | 0.412 | 0.178 | — |
| Q0-1 advantage-only | -0.331 | 0.317 | — |
| Q0-1 + uncertainty (selected) | **0.422** | **0.179** | 0.376 |
| Q0-1 + uncertainty + hard-negative weighting | 0.319 | 0.250 | 0.394 |
| Q0-2 local | 0.336 | 0.252 | — |
| Q0-3 Mini U-Net | 0.221 | 0.298 | — |
| Q0-3 patch 8 | 0.322 | 0.243 | — |
| Q0-3 patch 16 | 0.320 | 0.244 | — |
| Q0-4 joint error/advantage | 0.268 | 0.257 | — |
| Q0-5 uncertainty | 0.273 | 0.279 | 0.395 |

The explicit real hard-negative weighting was not silently discarded: it was
run after the main pilot and rejected because it worsened ranking. At the
MAE-selected epoch it reached top-1 0.180/regret 0.250; its best ranking epoch
still reached only 0.201/0.213, versus 0.270/0.179 without reweighting. The
tested implementation remains in `quality_losses.py` behind
`--hard-negative-weight`; the selected full checkpoint uses 0.0.

## Smoke and tests

- 27 deterministic tests pass.
- The 24-pair overfit smoke reduced total loss from 1.123 to -0.627 in 40
  updates. Training Pearson/Spearman reached 0.445/0.505, pairwise ranking
  accuracy 0.751, gradients remained nonzero, and all outputs stayed finite.
- The successful temporary smoke directory was deleted as required.

## Held-out seen results

### Absolute error prediction

| Candidate | MAE | Pearson | Spearman | Uncertainty/error Spearman | Interval calibration error |
|---|---:|---:|---:|---:|---:|
| raw | 0.565 | 0.617 | 0.200 | 0.489 | 0.0158 |
| t-1 | 0.560 | 0.603 | 0.249 | 0.521 | 0.0075 |
| t-2 | 0.564 | 0.604 | 0.273 | 0.510 | 0.0072 |
| t-4 | 0.576 | 0.626 | 0.349 | 0.488 | 0.0162 |
| t-8 | 0.611 | 0.695 | 0.490 | 0.419 | 0.0185 |

Pearson is positive for every candidate and all three backbones. The model is
biased low by roughly 0.31-0.49 px, however, and the low Spearman values for
raw/t-1 show that fine local ordering is substantially harder than detecting
large-error regimes.

### Diagnostic ranking (margin 0.10 px)

| Signal | Top-1 | Top-2 recall | Pairwise | Regret | Raw/null accuracy |
|---|---:|---:|---:|---:|---:|
| Constant train mean | 0.279 | 0.465 | 0.647 | 0.150 | 0.717 |
| FB/photo heuristic | 0.193 | 0.373 | 0.549 | 0.157 | 0.291 |
| Existing PPM selector | **0.283** | **0.500** | 0.609 | **0.119** | 0.448 |
| CMC spread | 0.112 | 0.219 | 0.595 | 0.154 | 0.366 |
| **Q0 quality predictor** | 0.197 | 0.362 | 0.548 | 0.153 | 0.458 |

Q0 fails the predeclared ranking gate: regret is 2.4% worse than the constant
baseline and 28.6% worse than the replayed PPM selector. Top-1 is 8.1 points
below constant and 8.6 points below PPM. Its raw-vs-best-memory AUROC/AP are
only **0.514/0.348**, so the explicit error maps do not yield a useful binary
intervention score either.

### Critical failure slices

| Slice | Prevalence | Q0 top-1 | Q0 regret | PPM regret |
|---|---:|---:|---:|---:|
| Minority memory correct | 12.9% | 0.086 | 0.280 | 0.194 |
| Correlated consensus wrong | 13.2% | 0.335 | 0.151 | 0.098 |
| Raw clean / plausible memory worse | 38.7% | 0.244 | 0.105 | 0.089 |
| Low FB but wrong memory | 43.5% | 0.249 | 0.195 | 0.127 |
| Low photometric but wrong memory | 28.4% | 0.265 | 0.193 | 0.136 |

The precise cases Q0 was meant to unlock remain worse than the existing PPM
scores. This is decisive evidence against promotion.

### Risk/coverage interpretation

Low predicted uncertainty does reduce regret (0.049 at 10% coverage versus
0.153 overall), but that 10% slice is 98.1% raw-clean and memory-action
precision is only 1.33%. It is therefore an easy/clean-pixel detector, not a
high-precision temporal-correction detector. Large predicted advantage raises
available oracle gain but also raises diagnostic regret sharply.

## Runtime

- Quality predictor: 0.270 ms/sample (4.33 ms per batch of 16).
- SEA-RAFT: 10.75 ms/sample in the final batched run.
- Total measured pipeline peak: 2.10 GiB GPU.
- Q0 parameters: 1,155.

## Interpretation

Explicit error prediction is partially learnable, especially for large errors
and older memories, and the Laplace uncertainty is well calibrated. But the
shared evidence mostly predicts *common difficulty*: candidates often rise and
fall together. It does not identify the candidate-specific residual needed to
rank a minority correct memory. Since it loses to both a constant policy and
the existing PPM logits on regret/top-1, Q0 does not justify Q1 selection.

## Exact commands

Tests:

```bash
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python -m pytest -q \
  model_design/tests/test_quality_predictor.py \
  model_design/tests/test_quality_prediction_dataset.py
```

Overfit smoke (temporary output removed after success):

```bash
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  scripts/run_quality_prediction.py --mode smoke \
  --output /tmp/argos_q0_smoke --architecture q0_5 \
  --target-mode uncertainty --ranking-weight 2.0 --steps 40 \
  --batch-size 24 --workers 0 --device cuda:0 --no-resume
rm -rf /tmp/argos_q0_smoke
```

Selected full seen training:

```bash
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  scripts/run_quality_prediction.py --mode train \
  --output results/quality_prediction --architecture q0_1 \
  --target-mode uncertainty --trainable-uncertainty \
  --ranking-weight 0.5 --uncertainty-weight 1 --hard-negative-weight 0 \
  --max-train-samples-per-sequence 128 \
  --max-validation-samples-per-sequence 200 --epochs 3 \
  --batch-size 16 --workers 16 --device cuda:0 --no-resume
```

Frozen held-out seen evaluation:

```bash
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  scripts/run_quality_prediction.py --mode evaluate \
  --output results/quality_prediction \
  --checkpoint results/quality_prediction/checkpoints/best_validation.pt \
  --max-validation-samples-per-sequence 300 --batch-size 16 \
  --workers 16 --evaluation-sample-pixels 256 --device cuda:0
```

The pilot commands and exact per-run configurations are preserved under
`pilot/*/config.json` and `pilot/*/run.log`; non-selected pilot checkpoints
were deleted. Only the selected best and final checkpoints remain.
