# Learned causal t-1 refiner validation

## Decision

**NO-GO for promoting this first learned t-1 model as the safe ARGOS v2
refiner. GO for the narrower hypothesis that a learned, backbone-agnostic
bounded residual can exploit causal BiDA evidence and generalize to an unseen
backbone.**

The selected 39,299-parameter A2 model improves all three seen backbones and
untouched Fast-FoundationStereo, recovering 32-41% of the raw-or-memory oracle
gain on seen backbones and 49% on the unseen backbone at coverage 0.50. However,
the selector has only about 0.55 AUROC, 1-2% recall at threshold 0.5, and a
29-30% clean-pixel degradation ratio. It therefore fails the explicit promotion
criterion requiring substantially better clean-input safety than the heuristic
gate. Longer memory remains deferred.

## Correctness and split

- Strictly causal pairs: t-1 -> t only; pairs never cross sequence boundaries.
- Train: all 13 accepted sequences from datasets 1, 2, 3, and 6.
- Validation/test: all four `dataset_7_keyframe_*` sequences, held out as one
  acquisition group.
- Training backbones, exactly balanced: S2M2-S, RAFT-Stereo, StereoAnywhere.
- Fast-FoundationStereo is absent from training, checkpoint selection,
  architecture selection, loss tuning, and the A2-A7 ablations.
- Primary cache-grid coverage is 0.50. Sensitivity is reported at 0.05, 0.25,
  0.50, and 0.90. On the held-out split, common support is 54.9%, 38.3%, 16.5%,
  and 0.35%, respectively; 0.90 is too sparse for an architecture decision.
- Native SCARED-C disparity is resized by coverage-normalized area averaging:
  `resize(disparity * valid) / resize(valid)`, followed by the `180/W_native`
  disparity scaling. This avoids invalid-zero attenuation near GT boundaries.
- All paired methods use the identical mask: GT coverage, raw validity, aligned
  memory validity, and warp support.
- Results are cache-grid metrics. No upsampled result is called native inference,
  and no true native backbone inference was run in this cache-only task.

The exact split is recorded in every `split_manifest.json`.

## Model and tensor contract

The fully convolutional model uses 3x3 convolutions, GroupNorm, SiLU, two small
residual blocks, and independent 1x1 heads:

```text
g_error  = sigmoid(head_error(features))
c_memory = sigmoid(head_memory(features))
d_ref    = d_raw + g_error * c_memory * 3px * tanh(delta)
```

The heads initialize to an exact identity output: `g_error ~= 0.018`,
`c_memory = 0.5`, and `delta = 0`. The update is bounded to +/-3 cache pixels.
The selected A2 variant uses current disparity, aligned t-1 disparity,
current-minus-memory signed/absolute disagreement, current validity, aligned
validity, and warp support. It receives no backbone identity or private stereo
features. A3 adds FB/flow evidence, A4 photometric evidence, A5 RGB, A6 clean
preservation, and A7 safety ranking. Fixed normalization is documented in the
model source.

## Stage 0 and overfit smoke

Final tests: **15 passed**. They cover causal order, no sequence crossing,
cache-to-GT mapping, coverage-normalized GT resize, deterministic validation,
identity initialization, bounded updates, refiner gradients, frozen inputs, the
frozen SEA-RAFT path, and the original BiDA warp equivalence suite.

The 24-pair S2M2-S overfit smoke reduced loss by 69.3%, moved the joint gate from
0.009 to 0.885, retained finite outputs and non-zero gradients, and reached a
maximum 2.94 px update under the 3 px bound. Its temporary output was deleted.
This is a pipeline/capacity check, not a scientific result.

## Ablation ladder

Common-pixel weighted results on 128 held-out pairs per sequence/backbone at
coverage 0.50:

| Variant | Added evidence/loss | Refined EPE | Oracle recovery | new-Bad3 | False update | Clean degradation |
|---|---|---:|---:|---:|---:|---:|
| A2 | disparity + validity | **0.7520** | **40.1%** | 0.033% | 32.2% | 26.5% |
| A3 | A2 + FB/flow | 0.7631 | 27.5% | 0.045% | 45.2% | 35.5% |
| A4 | A3 + photometric | 0.7576 | 33.7% | 0.038% | 55.3% | 38.7% |
| A5 | A4 + RGB | 0.7830 | 5.0% | 0.024% | 66.1% | 30.8% |
| A6 | A5 + clean loss | 0.7821 | 5.9% | **0.006%** | 42.3% | 32.8% |
| A7 | A6 + ranking loss | 0.7732 | 16.0% | 0.031% | 55.7% | 40.1% |

Raw EPE is 0.7874 and oracle EPE is 0.6991 for every row. A2 is selected entirely
from seen validation. More evidence does not monotonically help; in particular,
FB reliability and RGB do not improve this small selector, while the present
safety losses reduce new-Bad3 but not false updates or clean degradation.

## Final geometry

The final evaluation uses the frozen A2 best-validation checkpoint and 300 causal
pairs from each held-out sequence.

| Backbone | Raw EPE | Learned EPE | Oracle EPE | Learned gain | Oracle recovery |
|---|---:|---:|---:|---:|---:|
| S2M2-S | 0.6649 | **0.6459** | 0.6187 | 0.0190 | 41.1% |
| RAFT-Stereo | 0.5888 | **0.5630** | 0.5090 | 0.0258 | 32.3% |
| StereoAnywhere | 0.5896 | **0.5636** | 0.5151 | 0.0259 | 34.8% |
| Fast-FoundationStereo (unseen) | 0.6142 | **0.5929** | 0.5711 | 0.0212 | 49.3% |

Seen aggregate geometry at coverage 0.50 improves from 0.6144 to 0.5909 EPE,
Bad1 from 6.12% to 5.90%, Bad3 from 3.86% to 3.72%, and AbsRel from 0.0514 to
0.0496. Unseen Fast-FoundationStereo improves EPE, Bad1, Bad3, AbsRel, and
boundary EPE (0.7515 to 0.7232).

| Split | Coverage | Raw | Learned | Oracle | Oracle recovery | Common support |
|---|---:|---:|---:|---:|---:|---:|
| Seen | 0.05 | 0.5317 | 0.5204 | 0.4641 | 16.7% | 54.9% |
| Seen | 0.25 | 0.5094 | 0.4911 | 0.4455 | 28.6% | 38.3% |
| Seen | 0.50 | 0.6144 | 0.5909 | 0.5476 | 35.2% | 16.5% |
| Seen | 0.90 | 1.1122 | 1.0605 | 1.0220 | 57.3% | 0.35% |
| Unseen | 0.05 | 0.5278 | 0.5186 | 0.4761 | 17.8% | 54.9% |
| Unseen | 0.25 | 0.5034 | 0.4873 | 0.4584 | 35.9% | 38.3% |
| Unseen | 0.50 | 0.6142 | 0.5929 | 0.5711 | 49.3% | 16.5% |
| Unseen | 0.90 | 1.1083 | 1.0461 | 1.0299 | 79.3% | 0.35% |

At coverage 0.50 the model improves 11/12 seen backbone/sequence pairs; the sole
regression is S2M2-S on `dataset_7_keyframe_2` (+0.0031 px). It improves all four
Fast-FoundationStereo sequences. Exact values are in the sequence CSVs.

## Baselines and safety

| Method | Seen EPE | Unseen EPE |
|---|---:|---:|
| Raw | 0.6144 | 0.6142 |
| Memory replacement | 0.6150 | 0.6136 |
| Fixed blend 0.10 | 0.6112 | 0.6125 |
| Fixed blend 0.25 | 0.6083 | 0.6107 |
| Fixed blend 0.50 | 0.6068 | 0.6095 |
| Heuristic gate | 0.6070 | 0.6097 |
| Learned A2 | **0.5909** | **0.5929** |
| Oracle | 0.5476 | 0.5711 |

The masked GT resize changes absolute low-coverage baseline values relative to
the earlier BiDA probe; it removes invalid-zero attenuation and must not be mixed
with that report's unnormalized low-coverage EPE.

| Safety metric @ 0.50 | Heuristic seen | Learned seen | Heuristic unseen | Learned unseen |
|---|---:|---:|---:|---:|
| new-Bad3 | 0.124% | **0.043%** | 0.040% | **0.036%** |
| False update, clean pixels | **14.5%** | 28.5% | **13.5%** | 28.2% |
| Clean degradation | **19.2%** | 30.1% | **19.0%** | 29.1% |
| Mean clean update | **0.0325** | 0.0476 | **0.0274** | 0.0467 |
| Frames worsened | 34.9% | **30.5%** | 32.8% | **27.1%** |
| Worst frame degradation | 3.7094 | **0.1532** | **0.1169** | 0.1248 |
| p95 degradation | 0.0226 max | **0.0214 max** | 0.0174 | **0.0159** |

The learned model removes the heuristic gate's seen catastrophic tail and lowers
new-Bad3, frame-worsening rate, and p95 degradation. But it updates clean pixels
about twice as often and degrades a larger fraction of them. Safety is mixed.

Seen selector AUROC is 0.549-0.554 and AP 0.311-0.326. Precision at a 0.5 joint
gate is high (0.78-0.90), but recall is only 0.9-2.0%. On Fast-FoundationStereo,
AUROC is 0.554, AP 0.306, precision 0.741, and recall 1.13%. The residual extracts
useful geometry despite an under-confident, poorly discriminative selector; this
does not establish a reliable error detector.

## Runtime

| Evaluation | SEA-RAFT | Refiner | Total | Peak GPU memory | Parameters |
|---|---:|---:|---:|---:|---:|
| Seen | 2.66 ms/pair | 0.185 ms/pair | 2.85 ms/pair | 1.04 GB | 39,299 |
| Unseen | 3.23 ms/pair | 0.253 ms/pair | 3.48 ms/pair | 1.04 GB | 39,299 |

Times exclude stereo inference because disparity is loaded from validated caches.

## Reproduction commands

Tests:

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=1 .miniconda/envs/argos/bin/python -m pytest -q \
  ARGOS-V2/model_design/tests/test_bidavideo.py \
  ARGOS-V2/model_design/tests/test_learned_t1_refiner.py
```

Overfit smoke (temporary output deleted after success):

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=1 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_learned_t1_refiner.py --mode smoke \
  --output /tmp/argos_learned_t1_smoke --variant A7 --steps 60 --epochs 1 \
  --batch-size 8 --workers 8 --max-train-pairs-per-sequence 24 \
  --max-validation-pairs-per-sequence 4 --device cuda:0 --no-resume
rm -rf /tmp/argos_learned_t1_smoke
```

Single-backbone sanity:

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=1 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_learned_t1_refiner.py --mode train \
  --output ARGOS-V2/results/learned_t1_refiner/stage2_s2m2_masked \
  --variant A7 --backbones S2M2-S --epochs 3 --batch-size 32 --workers 16 \
  --max-train-pairs-per-sequence 128 --max-validation-pairs-per-sequence 64 \
  --coverage-threshold 0.50 --device cuda:0 --no-resume
```

Three-backbone A2-A7 training; each run is resumable. The actual jobs were split
across GPUs 0 and 1 and detached after their logs showed healthy execution.

```bash
for variant in A2 A3 A4 A5 A6 A7; do
  TMPDIR=/tmp CUDA_VISIBLE_DEVICES=1 .miniconda/envs/argos/bin/python \
    ARGOS-V2/scripts/run_learned_t1_refiner.py --mode train \
    --output ARGOS-V2/results/learned_t1_refiner/ablations/${variant} \
    --variant ${variant} --epochs 4 --batch-size 32 --workers 24 \
    --max-train-pairs-per-sequence 256 --max-validation-pairs-per-sequence 128 \
    --coverage-threshold 0.50 --device cuda:0 --resume
done
```

Final seen evaluation:

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=1 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_learned_t1_refiner.py --mode evaluate \
  --output ARGOS-V2/results/learned_t1_refiner/final_seen \
  --checkpoint ARGOS-V2/results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt \
  --backbones S2M2-S RAFT-Stereo StereoAnywhere \
  --validation-sequences dataset_7_keyframe_1 dataset_7_keyframe_2 \
    dataset_7_keyframe_3 dataset_7_keyframe_4 \
  --max-validation-pairs-per-sequence 300 --thresholds 0.05 0.25 0.50 0.90 \
  --batch-size 32 --workers 24 --device cuda:0 --contact-sheets 4
```

Frozen unseen evaluation:

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 .miniconda/envs/argos/bin/python \
  ARGOS-V2/scripts/run_learned_t1_refiner.py --mode evaluate \
  --output ARGOS-V2/results/learned_t1_refiner/final_unseen_fast_foundation \
  --checkpoint ARGOS-V2/results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt \
  --backbones Fast-FoundationStereo \
  --validation-sequences dataset_7_keyframe_1 dataset_7_keyframe_2 \
    dataset_7_keyframe_3 dataset_7_keyframe_4 \
  --max-validation-pairs-per-sequence 300 --thresholds 0.05 0.25 0.50 0.90 \
  --batch-size 32 --workers 24 --device cuda:0 --contact-sheets 4
```

## Interpretation

The central question receives a qualified positive: a tiny learned residual
captures a meaningful fraction of temporal oracle gain and transfers to the
primary unseen backbone. It does not yet learn a calibrated or sufficiently safe
memory selector. The next iteration, if pursued, should improve selector
supervision/calibration and clean-pixel intervention control on t-1. It should
not add t-2/t-4/t-8, PPMStereo, recurrence, Mamba, or pretrained RGB features.
