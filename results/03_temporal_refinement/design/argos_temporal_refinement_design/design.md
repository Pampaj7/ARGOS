# ARGOS Lightweight Temporal Stereo Refinement

## Goal

Build a small ARGOS-owned temporal refinement module that improves the temporal stability of frame-based stereo predictions without retraining S2M2.

The first target is:

- input baseline: `S2M2-L@736` for practical quality/speed, with `S2M2-S@512` as the deployment-fast variant;
- temporal teacher: `StereoAnyVideo@384x640`;
- output: refined disparity for center frame `t`, optionally with a confidence/failure map;
- first window: 5 frames, `t-2:t+2`;
- first data: SCARED warped/consecutive sequences, then GT keyframes where reliable.

This is an ARGOS module, not a modification of S2M2 or StereoAnyVideo.

## Why This Module

Current baseline facts:

- `S2M2-L@736` is the practical SCARED baseline.
- `S2M2-S@512` is the fastest baseline.
- `StereoAnyVideo@384x640` is smoother on the 32-frame SCARED sequence and is essentially tied with S2M2-L@full on the 5-frame GT smoke subset.
- StereoAnyVideo is too heavy to be the default deployment model, especially on longer windows, but it is a strong temporal teacher.

The useful research question is therefore: can a tiny temporal module recover some of StereoAnyVideo's temporal stability while keeping S2M2-L@736 or S2M2-S@512 runtime/VRAM close to deployable?

## Proposed Architecture Variants

### Variant A: Tiny 2D U-Net Residual Refiner

This should be the first prototype.

Input for center frame `t`:

- center RGB image: `I_t`, shape `[3, H, W]`;
- stacked S2M2 disparities over a 5-frame window: `[D_{t-2}, D_{t-1}, D_t, D_{t+1}, D_{t+2}]`, shape `[5, H, W]`;
- optional normalized disparity deltas: `[D_t - D_{t-1}, D_{t+1} - D_t]`, shape `[2, H, W]`;
- optional edge map from `I_t`, shape `[1, H, W]`;
- optional S2M2 failure mask, shape `[1, H, W]`.

Output:

- residual disparity `R_t`, shape `[1, H, W]`;
- refined disparity `D_ref_t = clamp(D_t + R_t, min=0)`;
- optional confidence/failure logit `C_t`, shape `[1, H, W]`.

Why first:

- simple;
- cheap;
- easy to debug;
- can be trained with small windows;
- naturally supports teacher and GT losses.

Recommended architecture:

- 3 encoder levels;
- depthwise separable or regular `3x3` convs;
- GroupNorm instead of BatchNorm;
- SiLU/ReLU activations;
- residual output initialized near zero;
- final residual scale clamp, for example `R_t = 4 * tanh(raw_R_t)` initially.

### Variant B: ConvGRU Temporal Refiner

Input:

- per-frame features from `[I_t, D_t, edge_t, optional failure_t]`;
- recurrent state propagated over the sequence;
- output refined disparity for each frame.

Advantages:

- more naturally video-native than Variant A;
- can run online;
- can use longer clips at inference.

Risks:

- harder to train stably;
- more sensitivity to sequence ordering and motion;
- needs careful state reset and masking around cuts.

Use after Variant A proves the loss/data path.

### Variant C: Low-Resolution Temporal Transformer

Input:

- low-resolution disparity/RGB features for 5 or more frames;
- temporal attention over patch tokens;
- upsample residual to full resolution.

Advantages:

- flexible temporal reasoning;
- can model longer interactions.

Risks:

- heavier;
- easier to overfit;
- less aligned with the "lightweight deployment" goal.

Use as a research branch, not the first implementation.

## Recommended First Prototype

Start with Variant A: Tiny 2D U-Net Residual Refiner.

### Input Tensor

For a 5-frame window at resolution `H x W`:

```text
rgb_center:        [B, 3, H, W]
s2m2_disp_window:  [B, 5, H, W]
edge_center:       [B, 1, H, W] optional
failure_prior:     [B, 1, H, W] optional
```

Concatenate as:

```text
x = concat(rgb_center_norm, disp_window_norm, edge_center, failure_prior)
```

First version channels:

```text
3 RGB + 5 disparity + 1 edge = 9 channels
```

If failure/confidence maps are added:

```text
10-11 channels
```

### Output Tensor

```text
residual_disp:     [B, 1, H, W]
refined_disp:      [B, 1, H, W]
confidence_logit:  [B, 1, H, W] optional
```

Train residual prediction, not absolute disparity, because S2M2 already gives a strong initialization.

## Training Data Plan

### Stage 0: Offline Cache

Create an ARGOS cache, ignored by Git:

```text
dataset/processed/temporal_refinement/
  scared_consecutive32/
    left/
    s2m2_l736_disp/
    s2m2_s512_disp/
    stereoanyvideo_disp/
    metadata.json
  scared_gt_keyframes/
    left/
    gt_disp/
    gt_depth/
    valid_mask/
    s2m2_l736_disp/
    stereoanyvideo_disp/
```

Cache S2M2 and StereoAnyVideo outputs once, then train the small refiner from cached tensors. This keeps training independent from S2M2/SAV runtime and avoids accidental retraining.

### Stage 1: SCARED Consecutive/Warped Sequences

Use:

- `ARGOS/dataset/scared_consecutive32`;
- any available SCARED warped/consecutive subsets;
- future extracted consecutive SCARED clips with GT if available.

Purpose:

- teacher distillation;
- temporal smoothness;
- flicker reduction.

### Stage 2: SCARED/SERV-CT GT Frames

Use reliable GT only:

- rectified SCARED keyframes with valid masks;
- SERV-CT CT-derived GT where geometry is trusted.

Purpose:

- prevent teacher drift;
- preserve metric accuracy;
- avoid learning only StereoAnyVideo bias.

### Stage 3: Mixed Surgical Training

Mix windows by source:

- 60-70% SCARED temporal windows;
- 20-30% GT supervised frames/windows;
- 10% hard cases from audits: specular, textureless, boundary, near-field.

## Loss Functions

Let:

- `D_ref_t`: refined disparity;
- `D_s2m2_t`: frozen S2M2 disparity;
- `D_sav_t`: StereoAnyVideo teacher disparity, resized/rescaled to original coordinates;
- `D_gt_t`: GT disparity;
- `M_gt`: valid GT mask;
- `E_t`: RGB edge/motion-aware mask.

### 1. Supervised Disparity Loss

Use where GT exists:

```text
L_gt_disp = Charbonnier((D_ref_t - D_gt_t) * M_gt)
```

Optionally add depth-space supervision:

```text
Z_ref = fx * baseline / max(D_ref_t, eps)
L_gt_depth = Charbonnier((Z_ref - Z_gt) * M_gt)
```

Depth loss should be clipped or robust because small disparity errors explode at distance.

### 2. Teacher Distillation Loss

Use StereoAnyVideo as temporal teacher:

```text
L_teacher = Charbonnier((D_ref_t - D_sav_t) * M_teacher)
```

Mask teacher loss where:

- teacher disparity is invalid/nonpositive;
- S2M2 and teacher strongly disagree near RGB edges;
- teacher output is obviously oversmoothed across surgical boundaries.

Start with low teacher weight, for example:

```text
lambda_teacher = 0.2 to 0.5
```

### 3. Temporal Smoothness Loss

Simple first version:

```text
L_temp = Charbonnier((D_ref_t - D_ref_{t-1}) * M_temp)
```

Where `M_temp` suppresses:

- strong RGB edges;
- large image changes;
- likely motion boundaries;
- invalid or extreme disparity pixels.

Better second version:

- estimate optical flow or use image-space correspondence;
- compare `D_ref_t` with warped `D_ref_{t-1}`.

Do not start with flow unless necessary.

### 4. Edge-Aware Preservation

Preserve detail from S2M2 around RGB edges:

```text
L_edge_preserve = Charbonnier((grad(D_ref_t) - grad(D_s2m2_t)) * M_edge)
```

Or penalize smoothing across image edges:

```text
L_smooth = exp(-alpha * |grad(I_t)|) * |grad(D_ref_t)|
```

This prevents the temporal module from becoming a blur filter.

### 5. Failure/Confidence Loss

If confidence/failure labels are available:

- pseudo-label failure where S2M2 error is high and GT exists;
- pseudo-label teacher disagreement zones;
- train confidence logit with BCE/focal loss.

Useful later for ARGOS-Fuse and unsafe-region reporting.

### Initial Loss Mix

For the first experiment:

```text
L = 1.0 * L_gt_disp
  + 0.2 * L_gt_depth
  + 0.3 * L_teacher
  + 0.1 * L_temp
  + 0.05 * L_edge_preserve
```

If training on no-GT temporal windows:

```text
L = 0.5 * L_teacher
  + 0.2 * L_temp
  + 0.05 * L_edge_preserve
  + 0.05 * L_anchor
```

Where:

```text
L_anchor = Charbonnier(D_ref_t - D_s2m2_t)
```

This prevents hallucinated teacher-only drift.

## Evaluation Metrics

Keep existing ARGOS metrics:

- disparity MAE/RMSE;
- Bad-1/2/3 px;
- depth MAE/median/RMSE;
- Bad-1/2/5 mm;
- invalid/failure ratios.

Temporal metrics:

- mean consecutive disparity difference;
- mean consecutive depth difference where metric conversion is available;
- per-pixel temporal standard deviation;
- temporal error variation where GT is available;
- edge-masked temporal difference.

Add one important diagnostic:

```text
accuracy_vs_flicker_delta
```

Report whether temporal stability improves without worsening depth MAE by more than a chosen tolerance.

## First Minimal Experiment

### Experiment Name

```text
argos_temporal_refiner_unet_s2m2l736_sav_teacher_v0
```

### Data

Use cached predictions from:

- `S2M2-L@736`;
- `StereoAnyVideo@384x640`;
- SCARED consecutive/warped clips;
- SCARED keyframe GT where available.

### Model

Tiny U-Net residual refiner:

- input: `[RGB_t, D_{t-2:t+2}, edge_t]`;
- output: residual disparity for `t`;
- resolution: start at 736-width aligned with S2M2-L@736;
- residual clamp: `[-4 px, +4 px]` initially.

### Training

- freeze S2M2 completely;
- use cached S2M2 predictions;
- no StereoAnyVideo backprop;
- train refiner only;
- batch size small, likely 1-4 windows;
- mixed precision allowed;
- early stop on validation flicker/accuracy tradeoff.

### Validation

Use:

- held-out SCARED keyframes with GT;
- held-out consecutive clip with temporal metrics;
- compare against S2M2-L@736 and StereoAnyVideo.

## Success Criteria

Minimum success:

- reduce mean consecutive disparity difference vs S2M2-L@736 by at least 20%;
- keep depth MAE within `+0.05 mm` of S2M2-L@736 on GT keyframes;
- do not increase Bad-2mm by more than 1 percentage point;
- runtime overhead below 20 ms/frame on RTX 3090 at 736-width;
- peak VRAM overhead below 1 GB beyond cached/inference tensors.

Strong success:

- approach StereoAnyVideo temporal smoothness within 20-30%;
- preserve or improve depth MAE vs S2M2-L@736;
- produce useful confidence/failure map correlated with GT errors;
- run in near real time with S2M2-S@512.

## Expected Risks

### Teacher Oversmoothing

StereoAnyVideo may smooth boundaries or small surgical structures. Edge-aware losses and residual anchoring to S2M2 are needed.

### No True GT On Consecutive Clip

Temporal metrics without GT can reward oversmoothing. Always pair temporal scores with GT keyframe accuracy.

### Non-Consecutive Keyframes

The 5-frame GT smoke sequence is not a true video. Do not use it for temporal claims.

### Scale/Resize Errors

Every teacher or S2M2 prediction must be rescaled back to original disparity coordinates:

```text
pred_disp_original = pred_disp_resized / scale_x
```

### Domain Shift

SERV-CT and SCARED geometry differ. Mixed surgical training should keep dataset tags and report cross-dataset behavior separately.

### Hidden Failure Modes

Specular highlights, tissue boundaries, low texture, blood, smoke, and tools may fool teacher and student differently. Add failure slices early.

## Reusable Code Structure

Recommended ARGOS-owned structure:

```text
scripts/temporal_refinement/
  README.md
  cache_predictions.py
  train_refiner.py
  eval_refiner.py
  models/
    tiny_unet_refiner.py
    convgru_refiner.py
  losses.py
  datasets.py
  metrics.py
  visualization.py

configs/temporal_refinement/
  unet_s2m2l736_sav_teacher.yaml
```

Keep model code generic:

- dataset returns tensors and masks;
- model only sees tensors;
- losses are composed from config;
- evaluation reuses existing ARGOS metric functions where possible;
- no dependency on upstream S2M2/StereoAnyVideo repos during training except for offline cache generation.

## Recommended Next Action

Implement the offline cache first.

Order:

1. Cache S2M2-L@736 predictions for SCARED consecutive clips.
2. Cache StereoAnyVideo@384x640 teacher predictions for the same clips.
3. Build a dataset class that yields 5-frame windows.
4. Implement Tiny U-Net residual refiner.
5. Run a 100-step overfit sanity test on one sequence.
6. Evaluate against S2M2-L@736 on temporal metrics and GT keyframes.

Do not start training until the cache and evaluation path are deterministic.

