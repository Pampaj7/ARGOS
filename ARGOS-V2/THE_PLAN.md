# THE PLAN - ARGOS v2 Temporal Stereo Refinement

## Project Status

This work must be developed, documented, and referred to as **ARGOS v2**.

Before modifying code, every agent must inspect:

```text
/dtu/p1/leopam/ARGOS/SOTA/
```

The `SOTA/` folder contains the current scientific plan, literature notes, technical decisions, and relevant architectural context. Agents must extract and use the relevant material from that folder before implementation.

## 1. Central Research Goal

ARGOS v2 aims to develop:

> A causal, safe, backbone-agnostic temporal stereo refiner for surgical video, trained across heterogeneous frozen stereo predictors and evaluated under unseen-backbone and cross-domain surgical shifts.

The refiner must improve temporal consistency without degrading already accurate stereo predictions.

The key scientific problem is not simply temporal smoothing. The real problem is:

> How can a temporal refiner determine when a stereo prediction is genuinely wrong, and when an apparent temporal change is caused by real motion, deformation, occlusion, or scene change?

The refiner must therefore learn when to intervene and when to preserve the raw prediction.

## 2. Motivation From ARGOS v1 Failure

Previous temporal refiners under `scripts/temporal_refinement/` failed under zero-shot out-of-domain evaluation.

On SERV-CT, raw stereo predictions were already highly accurate, but v1 refiners trained on SCARED-specific error distributions introduced severe false-positive corrections.

Observed failure:

| Method | EPE |
|---|---:|
| Raw stereo | ~1.28 px |
| MPC / CPV refined | ~6.3-6.6 px |

The refiner learned the error signature of:

```text
training backbone + training dataset
```

instead of learning a general temporal correction prior.

This failure defines the central design constraint of ARGOS v2:

> The temporal refiner must earn the right to modify the raw prediction.

## 3. Scientific Gap

Existing methods solve parts of the problem, but not the complete ARGOS v2 setting.

### EndoStreamDepth

Provides:

- causal streaming inference;
- hierarchical multi-scale temporal Mamba states;
- surgical-domain augmentations;
- endoscopic video depth validation.

Does not provide:

- plug-in refinement of arbitrary stereo backbones;
- backbone-agnostic inputs;
- unseen-backbone evaluation;
- explicit identity preservation;
- safety evaluation on already accurate predictions.

### BiDAStabilizer

Provides:

- plug-in refinement of frozen image-based stereo models;
- optical-flow alignment;
- residual disparity correction;
- forward and backward global propagation.

Does not provide:

- strict causal inference;
- bounded updates;
- explicit error gate;
- clean-input safety;
- multi-backbone training;
- unseen-backbone evaluation;
- surgical-domain validation.

### PPMStereo

Provides:

- reliability-aware selective memory;
- confidence, similarity, and redundancy-based memory selection;
- long-range temporal aggregation;
- top-K memory construction.

Does not provide:

- a plug-in refiner;
- backbone-independent inputs;
- cached disparity-only operation;
- strict causal streaming;
- identity-preserving safety;
- unseen-backbone evaluation;
- surgical-domain evaluation.

### ARGOS v2 Target

ARGOS v2 must combine:

```text
plug-in stereo refinement
+ causal streaming
+ selective temporal memory
+ multi-backbone training
+ unseen-backbone testing
+ surgical OOD testing
+ explicit identity preservation
```

## 4. Core Contribution

The intended main contribution is:

> ARGOS v2 introduces a causal, safe, and backbone-agnostic temporal stereo refiner trained across heterogeneous frozen stereo predictors. The model uses reliability-aware selective memory to improve temporal consistency while preserving accurate geometry, and is evaluated under unseen-backbone and surgical cross-domain shifts.

The contribution is not:

- "we use temporal memory";
- "we use Mamba";
- "we stabilize stereo video";
- "we add a ConvGRU";
- "we smooth disparity over time".

Those ideas already exist.

The novelty lies in the combination of:

- backbone-independent cached predictions;
- heterogeneous multi-backbone training;
- unseen-backbone generalization;
- causal selective memory;
- explicit clean-input safety;
- surgical cross-domain evaluation.

## 5. Datasets and Roles

### SCARED-C

Primary supervised temporal training dataset.

| Property | Value |
|---|---|
| Official size | 17,135 RGB-D pairs |
| Curated ARGOS v2 split | 16,921 valid frames, 17 accepted sequences |
| GT type | Metric temporal pseudo-GT derived from structured-light keyframes using corrected COLMAP/SfM poses and scale recovery |

Use for:

- training;
- validation;
- held-out temporal testing;
- multi-backbone refinement experiments.

### SCARED Structured-Light Keyframes

Direct geometric ground truth.

| Property | Value |
|---|---|
| Total keyframes | 45 |
| Clean keyframes | 35 from datasets 1, 2, 3, 6, 7 |

Use for:

- static geometry validation;
- checking whether the refiner damages direct structured-light geometry.

### SERV-CT

Contains:

- 16 stereo pairs;
- CT-derived geometry.

Use for:

- zero-shot static OOD evaluation;
- clean-input safety testing;
- reproducing the ARGOS v1 failure condition.

SERV-CT is especially important because raw stereo predictions may already be very accurate.

### D4D

Contains:

- long deformable stereo sequences;
- sparse Zivid structured-light anchors.

Use for:

- surgical OOD temporal evaluation;
- deformation robustness;
- anchor-based geometric validation.

### StereoMIS

Contains:

- long in-vivo stereo sequences;
- no dense depth ground truth.

Use for:

- zero-shot in-vivo temporal evaluation;
- no-reference temporal metrics;
- qualitative validation;
- optional later unsupervised adaptation.

StereoMIS adaptation is not part of the first milestone.

### C3VD / EndoSLAM / Hamlyn

Secondary datasets. Use only if needed for:

- additional surgical-domain analysis;
- monocular comparison;
- temporal no-reference analysis.

They are not primary training datasets for the first ARGOS v2 version.

## 6. Frozen Stereo Backbone Pool

Cache predictions from five heterogeneous stereo models.

### Selected Backbones

| Backbone | Family | Role |
|---|---|---|
| S2M2-S | Global scalable stereo matching; multi-resolution transformer-style matching | Main ARGOS anchor backbone |
| RAFT-Stereo | Iterative recurrent correlation; multi-level GRU refinement | Classical iterative baseline |
| StereoAnywhere | Stereo + monocular foundation prior fusion | Mono-stereo hybrid failure distribution |
| CREStereo | Cascaded coarse-to-fine recurrent matching; adaptive group correlation | Secondary recurrent architecture; ablation or second unseen backbone |
| Fast-FoundationStereo | Foundation-model stereo; distilled and deployment-oriented | Primary unseen backbone |

### Excluded Initially

#### DEFOM-Stereo

Excluded because of an unresolved evaluation bug:

- approximately 21.6 px on unified keyframes;
- worse than SGBM;
- possible inverse-depth scaling issue.

Do not include DEFOM predictions until the inference and evaluation pipeline is audited and corrected.

#### MonSter / MonSter++

Excluded initially because its mono-stereo fusion paradigm is partially redundant with StereoAnywhere.

#### Stereo Any Video

Do not use as a frame-wise training backbone. It is already a temporal stereo model and should be treated as an external temporal baseline, not as a source backbone for the refiner.

## 7. Cache Format

Canonical cache location:

```text
ARGOS-V2/cache_scaredc_backbones/<backbone>/<sequence_id>/
```

Each sequence must contain:

```text
disparity.npy
valid_mask.npy
frame_ids.npy
metadata.json
```

### Array Contract

| File | Shape | Dtype | Notes |
|---|---|---|---|
| `disparity.npy` | `[T, 144, 180]` | `float16` | C-contiguous, uncompressed, `np.load(..., mmap_mode="r")` |
| `valid_mask.npy` | `[T, 144, 180]` | `uint8` or `bool` | Prediction-valid mask only |
| `frame_ids.npy` | `[T]` | `int32` or `int64` | Exact source frame IDs |

### Canonical Disparity Convention

All caches must use:

- positive left disparity;
- disparity expressed in pixels at cache width 180;
- canonical resolution `144 x 180`.

When resizing from source width `W_source`:

```text
d_cache = resize(d_source, (144, 180)) * (180.0 / W_source)
```

Never resize disparity without scaling its magnitude.

### Metadata Requirements

Each `metadata.json` must record:

- backbone;
- checkpoint;
- sequence ID;
- frame count;
- source resolution;
- inference resolution;
- cache resolution;
- disparity convention;
- resize policy;
- invalid-value policy;
- runtime;
- git commit;
- dtype;
- shape;
- completion status.

### Cache Safety

The cache builder must support:

- atomic writes;
- resume;
- completion flags;
- frame-order validation;
- no silent overwrite of valid completed caches;
- separate prediction validity and GT validity.

Do not merge these into one irreversible mask:

```text
prediction_valid
gt_valid
training_valid
```

## 8. Cache Generation Plan

### Pilot First

Before full generation:

```text
2 representative SCARED-C sequences x 5 backbones
```

The pilot must verify:

- exact frame count;
- exact frame ordering;
- disparity sign;
- disparity scale;
- prediction statistics;
- valid ratio;
- float16 quantization error;
- projected storage;
- inference timing;
- random mmap loading;
- contact-sheet alignment.

Pilot outputs should include:

- `cache_integrity.csv`;
- `timing_summary.csv`;
- `storage_summary.json`;
- `float16_vs_float32_error.csv`;
- `random_access_benchmark.json`;
- diagnostic contact sheets.

### Full Run

Only after pilot promotion criteria are satisfied:

```text
17 sequences x 5 backbones
```

Expected storage: approximately 8.5 GB total.

### Compute Strategy

The DTU node has:

- 2 x NVIDIA H100 80 GB;
- powerful AMD EPYC CPUs;
- large RAM capacity.

Use parallel per-backbone dispatch across GPUs.

Do not artificially limit dataloader or preprocessing workers. Use many CPU workers for:

- image decoding;
- resizing;
- packing;
- validation;
- cache writing.

Keep smoke tests small.

## 9. Initial Experimental Split

The first serious experiment should use:

| Role | Backbones |
|---|---|
| Training | S2M2-S, RAFT-Stereo, StereoAnywhere |
| Primary unseen | Fast-FoundationStereo |
| Secondary unseen / ablation | CREStereo |

The fourth and fifth caches should remain available even if not used in the first training run.

## 10. Experimental Questions

ARGOS v2 must answer the following questions separately.

### Q1 - Seen-Backbone Improvement

Can one shared refiner improve multiple heterogeneous training backbones?

| Train | Test |
|---|---|
| S2M2-S, RAFT-Stereo, StereoAnywhere | Same three backbones on held-out SCARED-C sequences |

### Q2 - Unseen-Backbone Generalization

Can the refiner improve a backbone whose predictions were never seen during training?

| Train | Test |
|---|---|
| S2M2-S, RAFT-Stereo, StereoAnywhere | Fast-FoundationStereo |

The unseen backbone must not be used for:

- training;
- validation;
- checkpoint selection;
- architecture selection;
- loss tuning.

### Q3 - Cross-Domain Generalization

Can the frozen refiner transfer to unseen surgical datasets?

Test on:

- SCARED structured-light keyframes;
- SERV-CT;
- D4D;
- StereoMIS.

### Q4 - Clean-Input Safety

Does the refiner preserve already accurate predictions?

This is mandatory. The model must be evaluated specifically on:

- raw EPE below threshold;
- raw Bad3 approximately zero;
- clean pixels;
- clean frames;
- clean sequences.

### Q5 - Multi-Backbone Benefit

Does training on multiple backbones improve generalization compared with single-backbone training?

Compare:

```text
single-backbone refiner vs multi-backbone refiner
```

using the same architecture and split.

## 11. Architecture Direction

The model must not be a simple:

```text
concat disparities -> TinyUNet -> smoothed disparity
```

It must be designed around uncertainty, evidence, memory reliability, and safety.

Working name:

> Safe Causal Pick-and-Play Temporal Stereo Refiner

The architecture should contain five main parts:

1. universal evidence encoder;
2. causal motion alignment;
3. reliability-aware selective memory;
4. hierarchical temporal representation;
5. safe identity-preserving output.

## 12. Universal Evidence Encoder

The refiner must use only signals available for any frozen stereo backbone.

Potential inputs:

- `left_rgb_t`;
- `right_rgb_t`;
- `raw_disparity_t`;
- raw disparity gradients;
- stereo photometric residual;
- warped previous refined disparity;
- raw-current minus warped-history residual;
- warp validity;
- occlusion cues;
- previous memory reliability.

Optional backbone confidence may be used only as an auxiliary input with dropout.

The model must not require:

- internal stereo cost volumes;
- internal transformer features;
- backbone-specific hidden states;
- backbone identity;
- backbone-specific confidence format.

The initial model should not receive the backbone name.

## 13. Causal Motion Alignment

All temporal information must be aligned to the current frame before fusion.

For a past memory entry:

```text
memory_{t-k}
-> warp to frame t
-> validity and occlusion filtering
-> current-frame comparison
```

Only past frames may be used:

```text
t-1, t-2, ..., t-M
```

No future frame is allowed in the deployable model. The model must support true streaming inference.

## 14. Reliability-Aware Selective Memory

Do not indiscriminately average all previous states.

Each memory candidate should be scored using some combination of:

- estimated quality;
- similarity to current frame;
- warp validity;
- photometric consistency;
- stereo consistency;
- redundancy;
- age;
- historical confidence;
- occlusion likelihood.

Then select top-K memories, initially:

```text
K = 3 or 5
```

The validated BiDA probe changes the immediate order of work: alignment is a
GO as a universal evidence source, but direct t-1 replacement, fixed blending,
and hand-designed FB/photometric gates are NO-GO. First train and validate a
learned t-1 selector with an identity-preserving bounded residual. Use raw and
aligned disparity, signed/absolute disagreement, warp support, FB confidence,
photometric residual, flow magnitude, and RGB features. Supervise with the
per-pixel memory-vs-raw error target (hard margin or clipped continuous gain), and
report win/loss advantage distributions plus clean-pixel safety. Defer t-2/t-4/t-8
and PPMStereo-style multi-memory selection until this t-1 experiment captures a
substantial fraction of the oracle gain.

The first learned t-1 experiment has now been validated. A 39k-parameter
disparity/validity-only CNN improves all three seen backbones and untouched
Fast-FoundationStereo, recovering 32-41% and 49% of oracle gain respectively at
cache coverage 0.50. It is not promoted as the safe refiner: selector AUROC is
only about 0.55, recall is 1-2% at the nominal 0.5 gate, and clean-pixel
degradation remains about 29-30%. FB, photometric, and RGB additions did not beat
the minimal disparity evidence ablation, and the initial safety losses did not
control false updates. Continue t-1 selector calibration/safety work before any
longer-memory component.

The memory should include:

- short-term high-resolution evidence;
- longer-term coarse structural evidence.

Memory selection is inspired by PPMStereo, but must operate on universal plug-in signals rather than backbone-specific cost features.

## 15. Hierarchical Temporal Representation

The model should use temporal states at multiple scales.

| Scale | Role |
|---|---|
| `1/4` | Local edges, fine disparity changes, instrument boundaries |
| `1/8` | Mid-scale deformation, local geometry, motion context |
| `1/16` | Global shape, long-term memory, large structural consistency |

Possible temporal operators:

- Mamba;
- ConvGRU;
- gated recurrent state;
- temporal attention;
- hybrid recurrent + selective memory.

Do not claim novelty from using Mamba alone.

## 16. Safe Identity-Preserving Output

The model must predict separate quantities:

- error probability;
- correction proposal;
- correction confidence;
- maximum update magnitude;
- memory reset / trust-current signal.

Recommended output:

```text
d_refined = d_raw + g_error * c_memory * tau * tanh(delta)
```

Where:

| Term | Meaning |
|---|---|
| `g_error` | Probability that raw disparity is wrong |
| `c_memory` | Confidence in temporal evidence |
| `tau` | Bounded local update magnitude |
| `delta` | Proposed correction |

Initialization must satisfy:

```text
gate approximately zero
residual approximately zero
output approximately equal to raw
```

Identity preservation must be structural, not only loss-based.

## 17. Memory Reset and Forgetting

The model must be able to discard temporal history when it becomes unreliable.

Reset or reduce memory influence under:

- new occlusion;
- large camera motion;
- tissue deformation;
- instrument entry;
- specular reflection;
- motion blur;
- invalid warp;
- sudden photometric change;
- scene cut;
- low memory confidence.

The model must not blindly copy previous refined disparity.

## 18. Training Strategy

Each sampled temporal clip must come from one backbone only.

Do not change backbone between consecutive frames of the same clip. A batch may contain clips from different backbones.

Use uniform or explicitly balanced backbone sampling. Prevent one backbone from dominating due to:

- more valid pixels;
- more sequences;
- lower inference noise;
- higher cache coverage.

The first training pool is:

- S2M2-S;
- RAFT-Stereo;
- StereoAnywhere.

## 19. Backbone Dropout and Robustness

To encourage plug-and-play behavior:

- randomly remove backbone confidence;
- randomly reset memory;
- randomly start clips from the middle of a sequence;
- randomly drop auxiliary channels;
- randomly vary memory length;
- randomly perturb temporal spacing.

The model must not rely on one optional signal.

## 20. Synthetic Error Corruptions

Real backbone predictions should remain the main training source.

Optionally use a limited fraction of synthetic corruptions to widen the error distribution:

```text
70-80% real backbone predictions
20-30% controlled synthetic corruptions
```

Possible corruptions:

- local disparity flicker;
- temporary bias;
- edge bleeding;
- holes;
- disparity jumps;
- local blur;
- inconsistent occlusion;
- false confidence;
- clean-to-dirty transition;
- dirty-to-clean transition;
- short-lived systematic offset.

Synthetic corruptions must model plausible stereo failure modes, not generic Gaussian noise only.

## 21. Losses

Start with a controlled set of interpretable losses.

### Geometry Loss

```text
L_geo = robust_loss(d_refined, d_gt)
```

Use Charbonnier, Huber, or another robust regression loss.

### Error-Detector Loss

Supervise the error gate using raw prediction error.

Possible soft target:

```text
y_error = clip(abs(d_raw - d_gt) / sigma, 0, 1)
```

### Clean-Input Preservation Loss

For clean raw predictions:

```text
L_clean = 1[e_raw < epsilon] * abs(d_refined - d_raw)
```

This directly targets the ARGOS v1 failure.

### Safety Ranking Loss

Penalize refinement when it becomes worse than raw:

```text
L_safe = max(0, e_refined - e_raw + margin)
```

### Update Magnitude Loss

```text
L_update = abs(d_refined - d_raw)
```

This encourages minimal intervention.

### Temporal Error Consistency

Prefer stabilizing prediction error rather than raw disparity alone.

Avoid a loss that simply forces:

```text
d_t ~= warped d_{t-1}
```

because this can reward freezing and oversmoothing. Use motion-aware consistency and valid/occlusion masks.

### Edge Loss

Use edge-aware supervision where necessary to preserve:

- tissue boundaries;
- instrument edges;
- occlusion transitions;
- thin structures.

## 22. Metrics

### Spatial Geometry

Report:

- EPE;
- Bad1;
- Bad3;
- AbsRel;
- RMSE;
- depth MAE;
- delta1;
- boundary EPE;
- boundary F1.

### Temporal Consistency

Report suitable metrics such as:

- TEPE;
- delta temporal thresholds;
- motion-compensated temporal error;
- warp consistency;
- temporal error variance;
- flicker score.

Avoid relying only on raw frame-to-frame disparity variance.

### Safety Metrics

Mandatory:

- new-Bad3;
- false update rate;
- percentage of clean pixels degraded;
- mean update magnitude on clean pixels;
- percentage of frames worsened;
- refined minus raw EPE;
- worst-case degradation;
- 95th percentile degradation.

### Runtime

Report:

- refiner latency;
- total latency;
- FPS;
- GPU memory;
- parameter count;
- MACs or FLOPs.

The model must remain practical for surgical robotics.

## 23. Baselines

Minimum baselines:

- raw frozen backbone;
- EMA temporal smoothing;
- temporal median;
- flow-warp and blend;
- small ConvGRU;
- ARGOS v1 refiner;
- BiDAStabilizer-style baseline if feasible;
- Stereo Any Video as external full-model baseline.

A strong paper must demonstrate that gains do not come from simple smoothing.

## 24. Evaluation Matrix

### Experiment A - Debug / Sanity

Use:

- 1 sequence;
- 1 backbone;
- small frame subset.

Goal:

- verify loader;
- verify loss;
- verify alignment;
- verify gradients;
- overfit small subset.

### Experiment B - Single Backbone

| Train | Test | Purpose |
|---|---|---|
| S2M2-S | S2M2-S | Pipeline validation only |

Do not spend excessive project time here.

### Experiment C - Multi-Backbone Seen

| Train | Test |
|---|---|
| S2M2-S, RAFT-Stereo, StereoAnywhere | Same three |

### Experiment D - Unseen Backbone

| Train | Test |
|---|---|
| S2M2-S, RAFT-Stereo, StereoAnywhere | Fast-FoundationStereo |

### Experiment E - Secondary Unseen Backbone

Test on CREStereo.

### Experiment F - Cross-Domain Static OOD

Test on:

- SCARED structured-light keyframes;
- SERV-CT;
- D4D structured-light anchors.

### Experiment G - Cross-Domain Temporal OOD

Test on:

- D4D sequences;
- StereoMIS;
- SCARED original temporal sequences.

### Experiment H - Clean Prediction Safety

Evaluate specifically on:

- clean pixels;
- clean frames;
- clean sequences;
- SERV-CT cases where raw stereo is strong.

### Experiment I - Leave-One-Backbone-Out

Later rotate held-out backbones:

```text
train A+B+C -> test D
train A+B+D -> test C
train A+C+D -> test B
train B+C+D -> test A
```

One primary unseen experiment is sufficient for the first milestone.

## 25. Go / No-Go Criteria

### Cache Promotion

Proceed to full caching only if:

- frame IDs are exact;
- disparity scale is verified;
- sign is verified;
- prediction statistics are plausible;
- random mmap loading works;
- float16 error is negligible;
- projected disk use is below 15 GB.

### Architecture Promotion

Proceed beyond sanity testing only if:

- the model can overfit a small training subset;
- output initially equals raw prediction;
- gate values behave sensibly;
- loss terms are numerically stable;
- no frame-order leakage exists.

### Multi-Backbone Promotion

Proceed to unseen-backbone testing only if:

- at least two seen backbones improve;
- no seen backbone suffers severe degradation;
- safety metrics remain acceptable.

### OOD Claim

Do not claim OOD robustness unless:

- geometry is preserved on SERV-CT and D4D anchors;
- temporal metrics improve on D4D or StereoMIS;
- new-Bad3 and false-update rates remain controlled.

### Backbone-Agnostic Claim

Do not claim backbone agnosticism unless the refiner improves or safely preserves a completely unseen backbone.

## 26. Implementation Rules for Agents

Every agent must:

1. work under the name ARGOS v2;
2. inspect `/dtu/p1/leopam/ARGOS/SOTA/`;
3. inspect existing v1 code before rewriting;
4. reuse validated dataset loaders and OOD infrastructure when safe;
5. preserve existing reproducible results;
6. avoid deleting validated outputs without explicit justification;
7. write smoke tests;
8. log configuration, commits, checkpoints, and paths;
9. avoid hard-coded machine-specific assumptions;
10. keep cache conventions centralized;
11. verify disparity scaling numerically;
12. keep smoke tests small;
13. use many CPU workers when beneficial;
14. avoid artificially limiting workers on the AMD EPYC node;
15. design for resume and crash recovery.

## 27. Existing Infrastructure to Reuse

Inspect and reuse where appropriate:

```text
scripts/temporal_refinement/
scripts/temporal_refinement/ood/
scripts/temporal_refinement/ood/d4d/
results/03_temporal_refinement/
dataset/SCARED-C/
dataset/SCARED/
dataset/SERV-CT/
dataset/D4D/
dataset/StereoMIS/
```

The existing SERV-CT OOD pipeline is valuable because it reproduces the failure mode of v1.

Prefer adapting validated infrastructure over rewriting everything.

Rewrite only when:

- the old code encodes invalid assumptions;
- cache conventions are incompatible;
- temporal leakage exists;
- disparity scaling is ambiguous;
- the old implementation cannot support multi-backbone evaluation.

## 28. Immediate Next Steps

Current task:

> Complete and validate the five-backbone SCARED-C cache generation.

After cache completion:

1. audit all cache metadata;
2. build a unified mmap temporal dataset loader;
3. implement raw/EMA/median/flow-warp baselines;
4. implement safety metric suite;
5. reproduce ARGOS v1 degradation on SERV-CT;
6. create the first identity-preserving refiner;
7. overfit a small SCARED-C subset;
8. train on three backbones;
9. evaluate the unseen Fast-FoundationStereo backbone;
10. freeze the model and run surgical OOD tests.

## 29. First Architecture Prototype

The first serious prototype should include:

```text
universal stereo evidence encoder
+ causal aligned memory
+ reliability-aware top-K selection
+ multi-scale temporal state
+ error gate
+ bounded residual correction
+ memory reset signal
```

It should not initially include:

- backbone-specific features;
- full stereo re-estimation;
- future frames;
- large bidirectional transformers;
- diffusion;
- full backbone fine-tuning;
- StereoMIS adaptation;
- excessive architectural complexity without ablation value.

## 30. Final Success Condition

ARGOS v2 succeeds only if one frozen refiner can demonstrate:

```text
improvement on seen backbones
+ generalization to an unseen backbone
+ preservation of already clean predictions
+ zero-shot surgical cross-domain robustness
+ causal streaming operation
+ reasonable runtime
```

The central result should not be merely lower average error.

The central result should be:

> The model improves temporal stereo predictions when correction is justified, while remaining close to identity when the raw stereo geometry is already reliable.
