# BiDAVideo causal alignment validation

## Decision

**GO for BiDA alignment as a universal evidence channel; NO-GO for direct t-1
replacement, fixed blends, or the present hand-designed FB/photometric gate.** The
next refiner should learn an identity-preserving selector/residual gate over the
aligned memory and its evidence channels.

The true per-pixel oracle confirms complementary information in temporal memory
on all three sequences and all three backbones. However, the gain is modest on the
cache grid, forward-backward and photometric signals do not reliably identify the
useful pixels, and every untrained blend fails the clean-input safety criterion.
The requested t-2/t-4/t-8 study is deferred until learned t-1 selection is shown to
extract a meaningful fraction of the available oracle gain.

## Scope and correctness

- Strictly causal t-1 only: current frame and one past frame; no future frames.
- Backbones: S2M2-S, RAFT-Stereo, StereoAnywhere.
- Sequences selected by the existing quality-gate code, not by visual cherry
  picking: easy `dataset_7_keyframe_1`, difficult `dataset_2_keyframe_4`, and
  boundary-heavy `dataset_7_keyframe_3`.
- 300 contiguous frames per sequence, yielding 299 t-1 comparisons.
- SEA-RAFT and optical RAFT share identical frames, caches, GT and masks.
- Every method uses exactly
  `GT-valid & raw-valid & aligned-memory-valid & warp-support`.
- Cache and native namespaces are separate. Native disparity is upsampled to GT
  resolution and multiplied by `W_native/180` before direct comparison with the
  native disparity arrays returned by `scared_c_data.load_frame_gt`. Native rows
  are real per-pixel native-grid-from-cached-predictions evaluations, not rescaled
  cache scalar metrics and not true native-resolution backbone inference.
- Native validation uses 99 t-1 pairs per sequence/backbone (the first of the 100
  loaded frames has no predecessor). Flow is inferred at the canonical cache grid
  and resized to native dimensions with independent x/y component scaling.

## Main cache-grid result

The table is common-pixel-count weighted across 3 sequences and 3 backbones at
coverage threshold 0.25 (30,605,103 common pixel observations for SEA-RAFT).

| Flow | Method | EPE (px) | Delta vs raw | Frames worsened | New-Bad3 | Clean update (px) |
|---|---|---:|---:|---:|---:|---:|
| SEA-RAFT | raw | 5.3938 | 0 | 0% | 0% | 0 |
| SEA-RAFT | memory replacement | 5.4067 | +0.0129 | 51.5% | 6.17% | 0.1639 |
| SEA-RAFT | blend 0.10 | 5.3949 | +0.0011 | 51.2% | 0.79% | 0.0164 |
| SEA-RAFT | blend 0.25 | 5.3966 | +0.0029 | 51.4% | 1.97% | 0.0410 |
| SEA-RAFT | blend 0.50 | 5.3998 | +0.0060 | 51.5% | 3.56% | 0.0819 |
| SEA-RAFT | consistency-gated | 5.3986 | +0.0048 | 51.2% | 3.11% | 0.0652 |
| SEA-RAFT | per-pixel oracle | 5.3350 | **-0.0588** | 0% | 0% | 0.0352 |
| RAFT | raw | 5.3940 | 0 | 0% | 0% | 0 |
| RAFT | consistency-gated | 5.3991 | +0.0051 | 50.5% | 3.02% | 0.0634 |
| RAFT | per-pixel oracle | 5.3361 | **-0.0580** | 0% | 0% | 0.0328 |

SEA-RAFT oracle gain is 0.018-0.164 px depending on sequence/backbone; it is not a
single-sequence effect. Memory is better on 48.15% of common pixels overall, but
the average loss where it is worse exceeds the average gain where it is better.

Coverage threshold matters strongly:

| Cache GT coverage threshold | Mean common ratio | Common observations | Frames with evaluable EPE |
|---:|---:|---:|---:|
| 0.05 | 50.80% | 35,435,283 | 100% |
| 0.25 | 43.88% | 30,605,103 | 100% |
| 0.50 | 26.15% | 18,240,093 | 100% |
| 0.90 | 1.24% | 868,224 | 65.2% |

Threshold 0.90 is too sparse to serve as the sole architecture decision metric.

## Native-grid-from-cache validation (SEA-RAFT)

Common-pixel-count weighted across 3 sequences, 99 t-1 pairs each:

| Backbone | Raw EPE | Memory EPE | Gated EPE | Oracle EPE | Oracle gain | Gated frames worsened |
|---|---:|---:|---:|---:|---:|---:|
| S2M2-S | 7.4881 | 7.7286 | 7.5454 | 7.1811 | 0.3070 | 53.2% |
| RAFT-Stereo | 6.6813 | 6.8911 | 6.6814 | 5.8757 | 0.8056 | 41.4% |
| StereoAnywhere | 6.8995 | 7.1326 | 6.9242 | 6.1057 | 0.7939 | 46.5% |

These are not true native-resolution backbone results: cached 144x180 predictions
are upsampled and disparity-scaled to the GT grid. The large boundary-heavy errors
are reported rather than hidden. Per-sequence
native oracle gains range from 0.071 to 2.862 px, while memory-better ratios range
from 34.0% to 51.8%. The oracle remains complementary, but the simple gate does
not safely extract that gain.

The much larger EPE than the cache leaderboard is expected: that leaderboard
aggregates all 17 sequences (about 1.04 px), whereas this probe uses three
intentionally difficult sequences and 299 contiguous t-1 pairs. On the exact
native prefix used here, the ordinary cache-only evaluator gives 7.496, 6.688 and
6.904 px for S2M2-S, RAFT-Stereo and StereoAnywhere, respectively, matching the
raw rows above. This is a subset/mask/aggregation difference, not a disparity-sign
or GT interpretation error.

## Reliability and flow comparison

At cache threshold 0.25, memory advantage (`raw_error - memory_error`) is negative
in almost every populated FB-confidence and photometric-residual bin. For
SEA-RAFT FB-confidence bins [0,.25), [.25,.5), [.5,.75), [.75,1], weighted
advantages are +0.055, -0.047, -0.023 and -0.012 px; the only positive bin is
tiny (13,206 pixel observations). Photometric bins are likewise non-monotonic and
negative: -0.012, -0.050, -0.005 and -0.278 px. These signals do not yet predict
where memory is useful. Overall, memory wins on 48.15% of common pixels; its
conditional mean advantage on wins is about 0.122 px, while its conditional mean
loss on the remaining pixels is about 0.138 px. The win rate alone therefore hides
the asymmetric error distribution.

| Flow | Oracle gain | Memory-better | Support | FB-consistent | Photo residual | Flow latency | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| SEA-RAFT | 0.0588 px | 48.15% | 98.12% | 97.50% | 0.0809 | **5.91 ms** | **388 MB** |
| RAFT | 0.0580 px | 48.40% | 98.10% | **97.99%** | 0.0827 | 7.78 ms | 492 MB |

SEA-RAFT remains the primary implementation: effectively equal oracle utility,
slightly better support/photometric residual, and materially lower latency and
memory. RAFT's higher FB-consistency does not translate into better oracle gain
or safer gated fusion, confirming that flow selection cannot use FB error alone.

## Next learned experiment

Use a small universal evidence encoder on raw disparity, aligned t-1 disparity,
signed/absolute disagreement, warp support, FB error/confidence, photometric
residual, flow magnitude, and RGB features. Compare A0 raw, A1 fixed blends, A2
the heuristic gate, A3 a learned selector, A4 selector plus bounded residual, and
ablations A5 (without FB), A6 (without photometric residual), and A7 (without RGB).
Train the selector with `memory_error + margin < raw_error` or a clipped continuous
advantage target. Initialize the refiner near identity and bound its update as
`d_refined = d_raw + g_error * c_memory * tau * tanh(delta)`. Report oracle gain,
win/loss advantage distributions and quantiles, reliability-conditioned gains, and
clean-pixel safety. Only after t-1 succeeds should t-2/t-4/t-8 be evaluated.

## Reproduction commands

Unit/equivalence tests:

```bash
CUDA_VISIBLE_DEVICES=0 .miniconda/envs/argos/bin/python -m pytest -q \
  ARGOS-V2/model_design/tests/test_bidavideo.py
```

Exact smoke tests used (their temporary directories were removed after success):

```bash
CUDA_VISIBLE_DEVICES=0 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_bidavideo_validation.py \
  --output /tmp/argos_bida_smoke --backbones S2M2-S \
  --flow-models sea_raft --sequences dataset_7_keyframe_1 \
  --frames 5 --batch-size 2 --thresholds 0.25 0.90 \
  --native-frames 2 --device cuda:0 --no-resume

CUDA_VISIBLE_DEVICES=1 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_bidavideo_validation.py \
  --output /tmp/argos_bida_raft_smoke --backbones S2M2-S \
  --flow-models raft --sequences dataset_7_keyframe_1 \
  --frames 3 --batch-size 1 --thresholds 0.25 \
  --device cuda:0 --no-resume

rm -rf /tmp/argos_bida_smoke /tmp/argos_bida_raft_smoke
```

Full SEA-RAFT evaluation, including native validation:

```bash
CUDA_VISIBLE_DEVICES=0 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_bidavideo_validation.py \
  --output ARGOS-V2/results/bidavideo_validation \
  --backbones S2M2-S RAFT-Stereo StereoAnywhere \
  --flow-models sea_raft --frames 300 --batch-size 8 \
  --thresholds 0.05 0.25 0.50 0.90 --native-frames 100 \
  --device cuda:0 --resume
```

Full optical-RAFT comparator on the same 300-frame cache-grid subsets:

```bash
CUDA_VISIBLE_DEVICES=0 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_bidavideo_validation.py \
  --output ARGOS-V2/results/bidavideo_validation \
  --backbones S2M2-S RAFT-Stereo StereoAnywhere \
  --flow-models raft --frames 300 --batch-size 8 \
  --thresholds 0.05 0.25 0.50 0.90 --native-frames 0 \
  --device cuda:0 --resume
```
