# ARGOS v2 PPMStereo validation

## Decision

**NO-GO for adding PPMStereo-style long memory to the first ARGOS v2 refiner.**

This is not a claim that older frames contain no complementary disparity evidence. The exact opposite is observed: the true per-pixel BiDA-aligned oracle improves strongly when t-2/t-4/t-8 are added. The NO-GO is narrower and operational:

- fixed and deterministic top-K policies do not exploit the ceiling safely;
- the minimal learned selector improves raw disparity but is worse than the already validated learned BiDA t-1 refiner on every seen backbone and on unseen Fast-FoundationStereo;
- clean-input degradation and false updates remain too high;
- four-age SEA-RAFT increases temporal-flow latency by roughly 4x over t-1.

Therefore ARGOS v2 should retain the causal BiDA t-1 evidence path and improve its error/abstention safety before adding PPMStereo-style memory.

## Protocol

- Seen training backbones: S2M2-S, RAFT-Stereo and StereoAnywhere.
- Primary unseen: Fast-FoundationStereo, accessed only after checkpoint selection.
- Validation: all four complete `dataset_7_keyframe_*` sequences.
- 300 contiguous source frames per sequence; current frames 8 through 299 give 292 causal queries.
- Exact past ages: 1, 2, 4 and 8; no future access and no sequence crossing.
- Primary coverage: 0.50; oracle sensitivity is also reported at 0.05, 0.25 and 0.90.
- All primary paired comparisons use `GT coverage & raw-valid & aligned t-1 validity & t-1 warp support`. Older invalid candidates abstain to raw rather than shrinking the mask.
- Metrics are pixel-count weighted.
- Namespace: **cache-grid-from-cached-predictions**. No value in this report is native stereo inference or native-grid-from-cached-predictions.

The train split remains the existing deterministic ARGOS v2 split (datasets 1/2/3/6); all dataset 7 keyframes are held out. Exact records are in `split_manifest.json` and the learned selector manifest.

## Oracle horizon result

At coverage 0.50, across the three seen backbones:

| Ceiling | EPE | Gain over raw |
|---|---:|---:|
| Raw | 0.64384 | - |
| Raw-or-t1 oracle | 0.57495 | 0.06889 |
| Raw-or-any-memory oracle | **0.47672** | **0.16712** |

Long memory adds 0.09823 px beyond t-1. Incremental gains are 0.03491 from t-2, 0.03116 from t-4 and 0.03216 from t-8. The gain appears on every backbone and all twelve backbone/sequence groups. At the final oracle, the best choice is raw on 27.4% of pixels, age 1 on 18.7%, age 2 on 17.8%, age 4 on 17.8%, and age 8 on 18.4%.

The unaligned multi-memory oracle is even lower (0.41856). This does **not** establish that alignment is useless: an oracle can exploit accidental same-coordinate value matches that no deployable selector can identify. It does establish that oracle richness alone is insufficient evidence for a PPMStereo adapter. Actual fixed selectors are required for the alignment control.

## Unaligned versus BiDA-aligned fixed selectors

Seen aggregate, coverage 0.50:

| Method | EPE | Delta vs raw | Clean degradation | Frames worsened |
|---|---:|---:|---:|---:|
| Raw | 0.64384 | 0 | 0 | 0 |
| Existing learned t-1 A2 | **0.61971** | -0.02412 | 45.5% | 27.4% |
| Unaligned latest | 0.64084 | -0.00299 | 46.5% | 41.1% |
| Unaligned adapted PPM top-K=3 | 0.63598 | -0.00785 | 52.0% | 45.0% |
| BiDA uniform all four | 0.66773 | +0.02389 | 51.9% | 45.1% |
| BiDA recent K=3 | 0.64751 | +0.00367 | 47.8% | 39.3% |
| BiDA faithful-formula adapter K=3 | 0.64378 | -0.00006 | 47.5% | 38.6% |
| BiDA ARGOS deterministic K=3 | 0.64381 | -0.00003 | 47.4% | 38.7% |
| Multi-memory oracle | 0.47672 | -0.16712 | oracle | oracle |

K=1 is the latest-memory control; K=3 is evaluated explicitly. K=5 is not defined because the requested exact age set has only four candidates. The released PPMStereo score/read-out cannot be executed faithfully on heterogeneous caches: it requires learned context keys, cost-volume motion values, recurrent state and a PPMStereo confidence network. “Faithful-formula adapter” means only the released score/penalty/top-K mechanics with universal ARGOS evidence.

## Learned long-memory selector

The learned model has 43,412 parameters. It shares a small candidate encoder across ages, gives raw an explicit abstention logit, normalizes play weights across raw plus valid memories, aggregates aligned disparity, and applies the same identity-initialized bounded A2 residual form. It uses listwise best-candidate, selected-regret, geometry and existing safety losses.

| Backbone | Raw | Existing t-1 A2 | Learned long memory | Multi oracle |
|---|---:|---:|---:|---:|
| S2M2-S | 0.69611 | **0.67702** | 0.67913 | 0.57220 |
| RAFT-Stereo | 0.61744 | **0.59070** | 0.60420 | 0.41110 |
| StereoAnywhere | 0.61795 | **0.59143** | 0.60520 | 0.44686 |
| Fast-FoundationStereo unseen | 0.64515 | **0.62332** | 0.63578 | 0.52834 |

The long-memory model improves raw on all four backbones, but loses 0.00980 EPE to t-1 on seen aggregate and 0.01246 on Fast-FoundationStereo. It recovers only 8.6% of the multi-memory oracle gap on seen and 8.0% on Fast.

The selector does not collapse to t-1: seen mean play weights are 0.182/0.175/0.173/0.188 for ages 1/2/4/8, effective horizon 3.80. However, this is close to uniform rather than useful quality selection; argmax chooses age 8 on 30.5% of pixels and raw on 55.7%.

## Safety

| Evaluation | Clean degradation | False-update | Frames worsened | Worst frame | p95 frame |
|---|---:|---:|---:|---:|---:|
| Seen | 36.5% | 42.2% | 38.7% | +0.1230 | +0.0424 |
| Fast unseen | 35.7% | 41.7% | 41.0% | +0.0841 | +0.0400 |

This fails the promotion requirement even though mean EPE improves over raw. Exact per-frame and per-sequence values are retained in the compact CSV files.

## Runtime

- SEA-RAFT four-age flow: approximately 10.2 ms/current frame (four direct bidirectional candidate pairs).
- Evidence construction: approximately 1.5 ms/current frame in the oracle runner.
- Learned selector/refiner: 0.39 ms seen, 0.53 ms Fast.
- Oracle runner peak allocated GPU memory: 1,046 MB.
- Learned model: 43,412 parameters.

## Tests

The combined PPMStereo/BiDA/t-1 suite passes deterministic checks for causal ordering, reset, exact ages, no future/sequence crossing, tie-breaking, invalid exclusion, normalized weights, spatial redundancy, gradients, identity initialization, faithful original similarity, canonical BiDA reuse, no backbone input, and a synthetic known-best-memory case.

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 .miniconda/envs/argos/bin/python -m pytest -q \
  ARGOS-V2/model_design/tests/test_ppmstereo.py \
  ARGOS-V2/model_design/tests/test_bidavideo.py \
  ARGOS-V2/model_design/tests/test_learned_t1_refiner.py
```

## Exact experiment commands

From `/dtu/p1/leopam/ARGOS`:

```bash
# Stage 1 oracle horizon
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=1 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_ppmstereo_validation.py --mode oracle \
  --output ARGOS-V2/results/ppmstereo_validation \
  --backbones S2M2-S RAFT-Stereo StereoAnywhere \
  --sequences dataset_7_keyframe_1 dataset_7_keyframe_2 dataset_7_keyframe_3 dataset_7_keyframe_4 \
  --frames 300 --ages 1 2 4 8 --thresholds 0.05 0.25 0.50 0.90 \
  --batch-size 32 --device cuda:0 --resume

# Stage 2/3 fixed selectors (replace backbones with Fast-FoundationStereo for unseen control)
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=1 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_ppmstereo_baselines.py \
  --output ARGOS-V2/results/ppmstereo_validation \
  --backbones S2M2-S RAFT-Stereo StereoAnywhere \
  --sequences dataset_7_keyframe_1 dataset_7_keyframe_2 dataset_7_keyframe_3 dataset_7_keyframe_4 \
  --frames 300 --ages 1 2 4 8 --thresholds 0.05 0.25 0.50 0.90 \
  --batch-size 32 --device cuda:0 --resume

# Learned smoke (temporary output deleted after passing)
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_learned_ppm_selector.py --mode smoke \
  --output /tmp/argos_ppm_learned_smoke --steps 60 \
  --batch-size 8 --workers 4 --device cuda:0 --no-resume

# Three-backbone training
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=1 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_learned_ppm_selector.py --mode train \
  --output ARGOS-V2/results/ppmstereo_validation/learned_selector \
  --backbones S2M2-S RAFT-Stereo StereoAnywhere --ages 1 2 4 8 \
  --coverage-threshold 0.50 --max-train-samples-per-sequence 256 \
  --max-validation-samples-per-sequence 300 --epochs 4 \
  --batch-size 16 --workers 32 --learning-rate 0.002 --device cuda:0 --resume

# Frozen seen or unseen evaluation
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=1 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_learned_ppm_selector.py --mode evaluate \
  --output ARGOS-V2/results/ppmstereo_validation/learned_selector/evaluation_unseen_fast \
  --checkpoint ARGOS-V2/results/ppmstereo_validation/learned_selector/checkpoints/best_validation.pt \
  --backbones Fast-FoundationStereo \
  --validation-sequences dataset_7_keyframe_1 dataset_7_keyframe_2 dataset_7_keyframe_3 dataset_7_keyframe_4 \
  --max-validation-samples-per-sequence 292 --batch-size 16 --workers 32 \
  --coverage-threshold 0.50 --device cuda:0 --no-resume
```

The oracle run and fixed-baseline run are resumable per backbone/sequence. Only best and final learned checkpoints are stored. No dense disparity or flow cache was written.
