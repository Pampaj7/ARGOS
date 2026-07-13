# ARGOS v2 frozen DINOv3 feature validation

## Decision

**NO-GO for frozen DINOv3 ViT-L/16 as the representation for the current
temporal-memory selector.** Retain the validated learned BiDA t-1 A2 refiner and
do not add DINO runtime, feature caching, or DINO+PPM long memory.

This is not a claim that DINOv3 lacks useful dense features. It is the narrower
controlled result that its frozen RGB representation did not improve the
ARGOS-specific decision “is an aligned cached disparity more accurate than the
current cached disparity?”

The required stage gate stopped the study after frozen representation ranking.
No bounded DINO refiner was trained, no long-memory DINO+PPM model was built,
and Fast-FoundationStereo was never accessed. Geometric safety fields are
therefore explicitly null rather than reported as zero.

## Correctness and protocol

- Official DINOv3 repository commit:
  `346f38fee679c56a6888f91c51670fae61d364e0`.
- Official local checkpoint SHA-256:
  `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`.
- ViT-L/16: 24 blocks, width 1024, four storage tokens, 303,154,176 frozen
  parameters. Official strict checkpoint loading; no download fallback.
- ImageNet RGB normalization; native SCARED-C aspect ratio 4:5 is preserved.
  The tested inputs are 256x320 and 320x400, producing 16x20 and 20x25 patch
  maps with no crop, pad, or stretch.
- DINO layer indices are zero-based: early/intermediate 5, middle 11,
  intermediate-late 17, and late 23. P4-P6 fuse `(5,11,17,23)`.
- Past DINO maps use the canonical BiDA `resize_flow` and `causal_warp`; no
  second token-warp convention exists.
- Causal ages are exactly 1, 2, 4, 8. Samples do not cross sequences and never
  use future frames.
- Train backbones: S2M2-S, RAFT-Stereo, StereoAnywhere, balanced. Validation is
  on held-out dataset-7 keyframes 1-4. Training uses 624 examples (16 per train
  sequence/backbone); validation uses 288 (24 per held-out sequence/backbone).
- Primary GT coverage is 0.50. Memory usefulness requires a 0.05 cache-pixel
  advantage. The selector softmax is over raw/null plus all four memories.
- Every P0-P6 variant uses the same temporal samples, flow, split, targets,
  optimizer, five epochs, and 27,809-parameter shared ranker. P1/P5 additionally
  contain the explicitly reported 21,216-parameter RGB encoder. DINO channels
  are reduced by the same fixed seeded projection, so DINO does not win by a
  larger trainable decoder.

## Wrapper and test result

The canonical wrapper is
`model_design/external_components/dinov3.py::FrozenDINOv3`. It exposes
normalized intermediate patch maps, aspect-preserving preprocessing, BF16
autocast, latency/memory measurement, and canonical feature warping. It forces
eval mode and `torch.inference_mode` by default; every DINO parameter is frozen.

The combined DINO/BiDA/PPM tests pass: **38 passed** (13 DINO-specific plus the
existing BiDA and PPMStereo suites). Tests cover official checkpoint and
architecture, frozen/no-graph behavior, token reshape, deterministic layers,
x/y flow scaling, identity and integer feature warps, support, canonical BiDA
reuse, raw/null normalization, known-best targets, invalid exclusion, and
selector gradients without DINO gradients.

## Real-frame sanity and layer/resolution probe

The deleted tiny smoke used six real pairs spanning the four held-out sequences.
At 256x320 / 320x400, frozen DINO costs 28.72 / 28.74 ms per frame and peaks at
1.32 / 1.41 GB allocated GPU memory. These are isolated DINO timings, excluding
SEA-RAFT and selection.

Non-parametric cosine similarity alone does not identify memory usefulness:

| Resolution | Layer | memory-better AUROC | AP | best-memory top-1 | pairwise |
|---|---:|---:|---:|---:|---:|
| 256x320 | 5 | 0.491 | 0.266 | 0.329 | 0.574 |
| 256x320 | 11 | 0.478 | 0.253 | 0.328 | 0.575 |
| 256x320 | 17 | 0.468 | 0.245 | 0.324 | 0.572 |
| 256x320 | 23 | 0.457 | 0.242 | 0.324 | 0.570 |
| 320x400 | 5 | 0.495 | 0.267 | 0.327 | 0.577 |
| 320x400 | 23 | 0.456 | 0.240 | 0.324 | 0.570 |

The larger resolution increases four-layer FP16 storage from 2.50 to 3.91 MiB
per frame (+56.25%) for only +0.00093 AP and slightly worse top-1. The controlled
ranker therefore uses 256x320. No persistent DINO feature cache was built.

## P0-P6 controlled ranking

P0 is universal disparity/flow evidence; P1 is the local RGB CNN; P2 is DINO
late layer 23; P3 is DINO early/intermediate layer 5; P4 is four-layer DINO; P5
adds local RGB; P6 adds disparity/flow evidence.

| Variant | AUROC | AP | top-1 | pairwise | regret (lower) | null accuracy | null rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| **P0** | 0.614 | **0.222** | **0.677** | 0.317 | **0.107** | **0.693** | 0.954 |
| P1 | 0.534 | 0.180 | 0.653 | **0.457** | 0.117 | 0.683 | 0.909 |
| P2 | 0.587 | 0.187 | 0.624 | 0.248 | 0.141 | 0.653 | 0.877 |
| P3 | 0.607 | 0.197 | 0.656 | 0.332 | 0.110 | 0.677 | 0.930 |
| P4 | 0.549 | 0.175 | 0.621 | 0.444 | 0.129 | 0.655 | 0.876 |
| P5 | 0.530 | 0.175 | 0.587 | 0.381 | 0.121 | 0.635 | 0.819 |
| P6 | **0.618** | 0.214 | 0.666 | 0.434 | 0.117 | 0.687 | 0.928 |

P6's AUROC is 0.0038 above P0, but it loses 0.0074 AP, 0.0110 top-1 accuracy,
increases selected regret by 0.0104 cache pixels, and reduces raw/null accuracy
by 0.0061. P6 also selects memory more often without becoming more correct.

The result is consistent across all seen backbones:

| Backbone | P0 AP | P6 AP | P0 top-1 | P6 top-1 | P0 regret | P6 regret |
|---|---:|---:|---:|---:|---:|---:|
| S2M2-S | **0.218** | 0.212 | **0.683** | 0.674 | **0.078** | 0.085 |
| RAFT-Stereo | **0.226** | 0.216 | **0.664** | 0.647 | **0.115** | 0.134 |
| StereoAnywhere | **0.224** | 0.217 | **0.684** | 0.677 | **0.126** | 0.132 |

Thus DINO did not solve ranking, regret, calibration/abstention, or a hidden
single-backbone bottleneck. It also did not merely collapse all variants to the
same result: P1/P5 have larger trainable capacity and perform worse, supporting
the controlled representation conclusion.

## Stage gate, safety, and unseen protocol

Stage B was conditional on Stage A finding a DINO representation worth
integration. P0 remained best on the central metrics and on every seen
backbone, so training a DINO bounded residual would add a new intervention path
without representation evidence. Under PONYTAIL and the requested execution
order, the study stops here.

Consequently:

- t-1 DINO bounded-refiner result: **not run (failed Stage-A gate)**;
- long-memory DINO+PPM result: **not run (failed Stage-A and t-1 gates)**;
- Fast-FoundationStereo: **not accessed**; it remains untouched rather than
  being spent on a non-promoted configuration;
- false-update/new-Bad3/clean degradation: **not applicable, not zero** because
  Stage A ranks frozen representations and never updates disparity.

The validated BiDA t-1 A2 result and its existing unseen Fast-FoundationStereo
improvement remain the deployment reference. This experiment supplies no
evidence to replace it with DINO.

## Exact commands

```bash
cd /dtu/p1/leopam/ARGOS/ARGOS-V2

PYTHONPATH=. /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python -m pytest -q \
  model_design/tests/test_dinov3.py \
  model_design/tests/test_bidavideo.py \
  model_design/tests/test_ppmstereo.py

# Tiny smoke; /tmp/argos_dinov3_smoke was deleted after success.
PYTHONPATH=. /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  scripts/run_dinov3_feature_validation.py --mode smoke \
  --output /tmp/argos_dinov3_smoke --device cuda:0

PYTHONPATH=. /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  scripts/run_dinov3_feature_validation.py --mode probe \
  --output results/dinov3_feature_validation --device cuda:0 \
  --samples-per-sequence 24 --resolutions 256x320 320x400 \
  --layers 5 11 17 23

# Ranking smoke used 1 sample/sequence and 1 epoch; its /tmp output was deleted.
PYTHONPATH=. /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  scripts/run_dinov3_feature_validation.py --mode ranking \
  --output results/dinov3_feature_validation --device cuda:0 \
  --train-samples-per-sequence 16 --validation-samples-per-sequence 24 \
  --epochs 5 --batch-size 32 --learning-rate 0.002
```

Exact numeric data are in `feature_layer_probe.csv`, `ranking_metrics.csv`,
`aggregate_summary.json`, `safety_summary.json`, and `runtime_summary.json`.
