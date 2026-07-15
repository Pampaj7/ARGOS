# ARGOS v2 Q0 quality-prediction audit and predeclared protocol

## Scope

Q0 asks one question only: for the raw current disparity and the causally
BiDA-aligned disparities at ages 1, 2, 4 and 8, how large is the absolute error
likely to be, and how uncertain is that estimate? Q0 does not replace a
candidate, blend disparities, predict a residual, or implement Q1 selection.
The diagnostic `argmin(predicted error)` is used only to measure whether the
quality maps contain ranking information.

The evidence base is the completed ARGOS v2 BiDA, learned t-1, PPMStereo,
DINOv3, EndoStreamDepth and Cross-Memory Consensus studies. The relevant local
papers and notes under `SOTA/` were inspected together with `THE_PLAN.md`.

## Why explicit quality is the next controlled question

The learned t-1 memory-better head reached only about 0.55 AUROC and 1--2%
recall at its nominal gate, while its raw-error head had Brier 0.027--0.035 and
ECE 0.02--0.04. The PPMStereo universal selector spread weight almost uniformly
across ages and was worse than the t-1 model. DINOv3 did not improve ranking or
abstention. ConvGRU state was ignored. CMC's best train-sweep gain was only
0.0013 px and its ceiling captured about 30% of the multi-memory oracle because
errors are correlated and a useful candidate is often a minority witness.

These failures motivate predicting an absolute, calibrated quantity for every
candidate before asking a downstream policy to act. Consensus median/MAD remain
input evidence, never the decision.

## Candidate and tensor contract

Candidate order is immutable:

```
index:       0     1     2     3     4
candidate:  raw   t-1   t-2   t-4   t-8
age:         0     1     2     4     8
```

Dataset source tensors are `[K=5,C,H,W]`; batched tensors are
`[B,K,C,H,W]`, with `H=144`, `W=180`, positive-left disparity in cache-grid
pixels. Source frame IDs are exactly the current ID for raw and the IDs at the
four exact ages. Every record belongs to one sequence and one frozen stereo
backbone. RGB is loaded solely to run frozen SEA-RAFT/BiDA and is never passed
to Q0.

After alignment, each candidate exposes disparity, candidate validity, warp
support, FB error/confidence, photometric residual, flow magnitude and age. Raw
uses identity alignment: support/FB confidence equal raw validity; FB error,
photometric residual and flow magnitude are zero. Memory validity is exactly
`aligned_validity & warp_support`. No invalid candidate is assigned an error
target.

The shared Q0 feature tensor is `[B,K,F,H,W]` and contains only normalized
universal signals: raw/candidate disparity, signed/absolute disagreement,
candidate disparity gradients, local variance, raw/candidate validity, warp
support, FB error/confidence, photometric residual, flow magnitude, age,
cross-memory median/MAD/count, and candidate deviation from consensus. It has
no RGB, backbone identity, cost volume, stereo feature, hidden state or future
frame.

## Ground truth, coverage and masks

Native SCARED-C disparity and validity are resized using the already validated
coverage-aware rule:

```
coverage = area_resize(valid)
gt_cache = area_resize(disparity * valid) / max(coverage, 1e-6)
gt_cache *= 180 / W_native
```

At coverage threshold `q`, candidate `i` is supervised on:

```
target_valid_i = (coverage > q) & raw_valid & candidate_valid_i & warp_support_i
error_i        = abs(candidate_disparity_i - gt_cache)
advantage_i    = error_raw - error_i
```

For raw, `candidate_valid_0 = warp_support_0 = raw_valid`. Regression uses each
candidate's own target mask. A pairwise ranking comparison uses the intersection
of both candidate masks. Dataset-level paired method comparisons use identical
masks. Invalid candidates are excluded, never encoded as zero-error labels.
Primary model selection uses `q=0.50`; sensitivity is reported at 0.05, 0.25
and 0.90.

For regional targets, masked means are computed independently in 8x8 or 16x16
cells. A cell is valid only when it has supervised pixels. Median error,
memory-better fraction, clean-pixel fraction and boundary fraction are metrics,
not substitute regression labels.

## Outputs and target variants

The network produces `[B,K,H,W]` maps:

```
mu    = softplus(raw_mu)
sigma = softplus(raw_sigma) + 1e-3
advantage = raw_advantage_head
```

The predeclared pilot variants are:

- Q0-A: robust absolute-error regression;
- Q0-B: robust `log(error + 1e-3)` regression through `log(mu + 1e-3)`;
- Q0-C: raw-relative advantage regression;
- Q0-D: joint absolute error and advantage;
- Q0-E: absolute error plus heteroscedastic Laplace likelihood.

Losses are introduced in order: error only; error+ranking;
error+uncertainty; error+ranking+uncertainty; error+advantage+uncertainty.
Sigma is bounded inside the likelihood for numerical stability and receives an
explicit high-sigma penalty; this prevents trivial inflation without changing
the positive output contract.

Ranking ignores candidate pairs whose true-error difference is at or below the
indifference margin. Sensitivity margins are 0.05, 0.10, 0.25 and 0.50 px; no
margin may be tuned on an unseen backbone.

## Split and leakage policy

The deterministic learned-t1 split is reused unchanged. Training uses complete
accepted sequences outside dataset 7 and the balanced seen pool S2M2-S,
RAFT-Stereo and StereoAnywhere. All four dataset-7 keyframes are held out.
Fast-FoundationStereo and CREStereo are rejected by the Q0 dataset and runner
for training, validation, architecture/loss/patch selection, diagnostics and
checkpoint selection. No unseen evaluation is authorized in Q0.

## Metrics

Error prediction: MAE, RMSE, Huber, bias, Pearson, Spearman, explained variance,
true-error quantiles, candidate, age, sequence and backbone. Uncertainty:
Laplace NLL, calibration error, uncertainty/error correlation, sharpness and
empirical interval coverage. Ranking: top-1, top-2 recall, valid-pair accuracy,
normalized diagnostic regret, raw-vs-best-memory AUROC/AP, raw/null accuracy and
the 5x5 confusion matrix. Risk-coverage is reported at 1, 5, 10, 20, 50 and
100%, using predicted uncertainty (and separately predicted advantage where
available).

Hard-case slices are predeclared: memories agree but are jointly wrong; raw is
clean while plausible memories are worse; minority memory beats both raw and
consensus; low FB error but wrong memory; low photometric residual but wrong
memory; boundaries. Sampling may rebalance those measured real cases but may
not fabricate labels.

## Baselines and fair comparisons

On the same Q0 records and masks, compare train-set per-candidate mean error,
the learned t-1 raw/error and memory gates, the learned PPMStereo logits, and
CMC median/MAD/deviation proxies. Previously published summaries remain context
only when an exact same-subset replay is unavailable; numbers from different
subsets are not presented as paired improvements.

## Predeclared promotion gate

Q0 is GO only if the selected configuration, on held-out seen sequences:

1. improves top-1 accuracy over the strongest replayed selector signal by at
   least 3 percentage points or reduces normalized regret by at least 15%;
2. has positive Pearson and Spearman correlation on all three seen backbones,
   with aggregate Spearman at least 0.25;
3. has uncertainty/error Spearman at least 0.20 and 10%-coverage risk at most
   70% of full-coverage risk;
4. does not fall more than 2 top-1 points below the constant baseline on any
   seen backbone;
5. improves either minority-correct-memory or correlated-consensus-failure
   regret by at least 10% relative to the strongest baseline;
6. obtains the result with non-sparse valid support at primary coverage 0.50.

Otherwise Q0 is NO-GO. A visually plausible error map, a lower training loss,
or performance only at coverage 0.90 is insufficient.
