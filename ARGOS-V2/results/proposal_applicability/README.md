# ARGOS v2 — Proposal Applicability

## Verdict

**NO-GO for promotion.** Predicting the utility of the specific frozen A2
proposal is more discriminative than using the Raw Error Detector as a proxy,
and the selected P4 model is safer.  It does not, however, retain the
predeclared 70% of the existing authorized-A2 gain.

On final held-out SCARED-C keyframes 3/4 at cache-grid coverage 0.50:

| Method | EPE | Gain | Coverage | Precision | False update | Clean degradation |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 0.961605 | — | 0% | — | 0% | 0% |
| Frozen A2, unconditional | 0.918975 | 0.042630 | 39.96% | 61.87% | 34.14% | 12.96% |
| Existing Raw Error authorization | 0.934660 | 0.026946 | 5.62% | 75.63% | 1.90% | 0.98% |
| **P4 proposal authorization** | **0.948138** | **0.013467** | **3.18%** | **84.11%** | **1.19%** | **0.40%** |
| Update-magnitude baseline | 0.931386 | 0.030219 | 3.98% | 78.79% | 0.97% | 0.66% |
| Oracle proposal authorization | 0.913909 | 0.047696 | 10.77% | 100% | 5.65%* | 0% |

`*` The oracle's “false update” count uses the historical clean-pixel/update
definition: a clean pixel can still receive a genuinely beneficial proposal.
Its clean degradation is exactly zero.

P4 recovers 28.2% of the oracle authorization gain and 50.0% of the existing
Raw Error authorized gain.  It improves all three seen backbones, but this is
below the required 70%.  Consequently Fast-FoundationStereo, CREStereo,
SERV-CT, D4D, structured-light, and StereoMIS were not loaded.

## Target audit

The pre-implementation audit used 1,248 causal training pairs: 32 pairs from
each of 13 sequences for each of S2M2-S, RAFT-Stereo, and StereoAnywhere.  The
common mask contained 8,948,529 pixels.

- Mean utility: +0.0190 px; standard deviation: 0.0950 px.
- Pearson correlation between raw error and utility: 0.528 (sampled Spearman:
  0.341).  Utility is therefore not equivalent to raw-error detection.
- At epsilon 0.10: 8.46% helpful, 1.93% harmful, 89.60% indifferent.
- A2 harms by more than 0.10 px on 1.40% of raw-wrong pixels.
- A2 helps by more than 0.10 px on 5.93% of raw-clean pixels.
- Utility correlation is strongest with A2 update magnitude (Pearson 0.612),
  followed by raw error (0.528); photometric and FB signals are nearly
  uncorrelated.

Full per-backbone/per-sequence and epsilon statistics are in
`target_statistics.csv` and `target_correlations.csv`.

## Frozen input and model

P4 receives 23 backbone-independent cache-grid maps: raw, aligned t-1 and A2
disparities; signed/absolute A2 update; signed/absolute raw-memory disagreement;
A2 error gate, memory confidence and pre-tanh delta; three validity/support
masks; raw/A2/update x/y gradients; flow magnitude; photometric residual; FB
error and confidence.  It receives no GT, identity, RGB, future frame, stereo
feature, or cost volume.

The selected P4 is three 3x3 convolutions with 24 channels and utility,
uncertainty, and three-class heads: 15,533 trainable parameters and 8x8
end-to-end support including explicit gradient evidence.  Output utility is
bounded to [-3,3] px.  Authorization is frozen to:

```text
predicted utility > 0
and predicted class == helpful
and sigma < 2 px
and valid BiDA support
and abs(A2 update) <= 3 px
```

Rejected pixels equal raw bit-exactly; accepted pixels equal frozen A2 exactly.

## Controlled P1–P4 ladder

| Variant | Params | RF | Calibration AP | P0 proxy AP | Utility MAE | Intervention precision | Harmful acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 utility, 1x1 | 1,201 | 2 | **0.599** | 0.414 | **0.0358** | 44.10% | 46.90% |
| P2 local utility | 15,433 | 8 | 0.467 | 0.414 | 0.0480 | 38.86% | 55.09% |
| P3 local + uncertainty | 15,458 | 8 | 0.454 | 0.414 | 0.0360 | 48.57% | 47.50% |
| P4 + three classes/abstention | 15,533 | 8 | 0.567 | 0.414 | 0.0456 | **68.42%** | **10.16%** |

P1 shows that continuous utility is learnable, but its operating point is
unsafe.  P2 does not outperform P1: local spatial context is not the missing
factor in this controlled ladder.  P4 was frozen because it materially lowers
harmful acceptance while preserving AP above the P0 proxy.

Loss weights were: utility Huber 1.0 for all variants; P3/P4 Laplace
heteroscedastic 0.05; P4 weighted three-class CE 0.20 and explicit
harmful-as-helpful probability penalty 0.50.  Training pixels were
deterministically balanced over class, raw-error bin, update bin, and boundary;
records were balanced by backbone/sequence.  Calibration and test retain the
natural distribution.

## Final proposal diagnostics

At epsilon 0.10 on keyframes 3/4, P4 helpfulness AUROC/AP are
0.9576/0.7286 versus 0.8759/0.5760 for the Raw Error proxy.  P4 utility MAE is
0.0893 px, Pearson 0.588, Spearman 0.118, and uncertainty/error correlation
0.834.  Its sampled harmful-proposal acceptance is 10.18%, versus 29.51% for
the Raw Error authorization.

Safety improves in the severe tail: new-Bad3 is 0.0031%, worst-frame
degradation 0.0187 px, and mean clean update 0.00255 px.  However 4.58% of
frames worsen versus 3.33% for the existing authorizer, and the lower coverage
loses half of the existing EPE gain.  The simple update-magnitude comparator is
also stronger geometrically, so the proposal detector is not justified as the
promoted authorization policy.

## Split, masks, and leakage

- Train: 13 accepted non-dataset-7 sequences, three seen backbones.
- Calibration/selection: dataset_7 keyframes 1/2 only.
- Final seen: dataset_7 keyframes 3/4, opened after P4/checkpoint/epsilon/threshold freeze.
- Primary common mask: GT coverage >0.50, raw valid, aligned valid, warp support.
- Every compared method uses the same mask and cache-grid pixel units at width 180.
- Sensitivity outputs are present for coverage 0.05/0.25/0.50/0.90.
- Frozen A2 SHA256: `6cd29277397001333ef3ce630b2f3bc04ec393cdc72e65aa5eb087afd3b389ea`.
- Selected P4 SHA256: `c4d7d732b44ede1bb831b7789d6791907412b99345105f998c49c1cecde5bd2b`.
- No unseen backbone or OOD loader exists in this runner; promotion failure
  blocks the guarded unseen mode.

## Smoke, tests, and runtime

The real 24-pair P4 smoke reduced loss by 67.6%, remained finite, reached
utility Spearman 0.300 and helpfulness AP 0.525, then its temporary directory
was deleted.  The retained compact evidence is `smoke_summary.json`.

The focused suite passes 33 deterministic tests (plus the canonical BiDA tests)
covering target arithmetic, epsilon labels, paired masks, causal records,
stratified determinism, no identity/GT inputs, positive uncertainty, bounded
updates, detector-only gradients, frozen A2/SEA-RAFT, bit-exact abstention,
exact A2 acceptance, checkpoint determinism, and absence of OOD/dense-cache
paths.

P4 latency is 0.182 ms/frame at batch 16 on H100.  The complete frozen
evaluation allocated 598 MiB peak GPU memory (including SEA-RAFT/A2); P4 has
15,533 parameters.  Detector overhead remains negligible relative to the
validated flow/evidence path.

## Exact commands

```bash
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
cd /dtu/p1/leopam/ARGOS/ARGOS-V2

$PY -m pytest -q \
  model_design/tests/test_proposal_utility_dataset.py \
  model_design/tests/test_proposal_applicability_detector.py \
  model_design/tests/test_bidavideo.py \
  model_design/tests/test_learned_t1_refiner.py::test_cache_gt_is_coverage_normalized

CUDA_VISIBLE_DEVICES=0 $PY scripts/run_proposal_applicability.py \
  --mode smoke --output /tmp/argos_proposal_applicability_smoke \
  --variant P4 --channels 24 --batch-size 8 --workers 8 \
  --learning-rate 0.003 --device cuda:0

# P1-P4 used identical options and two independent H100s in pairs.
CUDA_VISIBLE_DEVICES=<0-or-1> $PY scripts/run_proposal_applicability.py \
  --mode train --output results/proposal_applicability/<P1-P4> \
  --variant <P1-P4> --channels 24 --epochs 5 --batch-size 16 \
  --workers 32 --max-train-pairs 256 --max-validation-pairs 160 \
  --learning-rate 0.002 --device cuda:0

CUDA_VISIBLE_DEVICES=<0-or-1> $PY scripts/run_proposal_applicability.py \
  --mode calibrate --output results/proposal_applicability/<P1-P4> \
  --checkpoint results/proposal_applicability/<P1-P4>/checkpoints/best_validation.pt \
  --variant <P1-P4> --batch-size 16 --workers 32 \
  --max-validation-pairs 160 --sample-pixels-per-frame 2048 --device cuda:0

# Executed only for frozen P4 after selection on keyframes 1/2.
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_proposal_applicability.py \
  --mode evaluate --output results/proposal_applicability/P4 \
  --checkpoint results/proposal_applicability/P4/checkpoints/best_validation.pt \
  --variant P4 --batch-size 16 --workers 32 \
  --max-validation-pairs 160 --sample-pixels-per-frame 2048 --device cuda:0
```

No unseen or OOD command was executed because the seen promotion gate failed.
No prediction, flow, or per-pixel feature cache was written.
