# ARGOS v2 raw-error detection and abstention audit

## Scientific scope

This study asks only whether a small universal detector can authorize the
already validated causal BiDA t-1 A2 update safely. It does not learn a new
correction, select among long memories, or jointly fine-tune A2. SEA-RAFT,
stereo caches and the A2 checkpoint remain frozen.

The relevant material under `SOTA/`, `THE_PLAN.md`, and the completed BiDA,
learned-t1, quality-prediction, PPMStereo and consensus results was inspected.
The validated evidence is consistent: aligned t-1 contains complementary
geometry and A2 improves all seen backbones, while candidate-relative scoring
is unreliable. Q0 nevertheless showed that absolute raw-error magnitude is
more predictable than relative memory correctness.

## Frozen correction reference

The proposal is produced by
`model_design/models/learned_t1_refiner.py::LearnedT1Refiner`, variant A2,
loaded from
`results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt`.
Its output is exactly:

```
u_A2 = g_error_A2 * c_memory_A2 * 3 px * tanh(delta_A2)
d_A2 = d_raw + u_A2
```

At cache coverage 0.50 on the previous four-sequence seen evaluation, A2 raw
EPE -> A2 EPE was 0.845 -> 0.818 (S2M2-S), 0.753 -> 0.715
(RAFT-Stereo), and 0.764 -> 0.723 (StereoAnywhere). Its clean false-update
rate was 31.6--32.6% and clean degradation 26.3--26.9%. Those are the safety
figures the authorization layer must reduce, not a new correction baseline.

## Data, split and causality

The validated `TemporalPairDataset` supplies exact t-1 -> t pairs on the
144x180 cache grid. Training uses all accepted non-dataset-7 sequences and the
balanced seen pool S2M2-S, RAFT-Stereo and StereoAnywhere. Dataset-7 remains
group held out and is subdivided before any run:

- calibration/model-selection: dataset_7_keyframe_1 and keyframe_2;
- frozen seen test: dataset_7_keyframe_3 and keyframe_4.

Fast-FoundationStereo and CREStereo are rejected before cache loading during
training, architecture/loss selection, calibration, threshold selection and
seen testing. They may be touched only after a seen GO. Every record retains
backbone, sequence and exact past/current frame IDs; no future frame exists in
the interface.

Native SCARED-C disparity is resized by the validated rule:

```
coverage = area_resize(valid)
gt_cache = area_resize(disparity * valid) / max(coverage, 1e-6)
gt_cache *= 180 / W_native
```

## Targets and masks

For cache-grid pixel `p`:

```
e_raw = abs(d_raw - d_gt)
y_epsilon = 1[e_raw > epsilon]
regression_valid = (coverage > q) & raw_valid
classification_valid = regression_valid & (abs(e_raw - epsilon) > band)
```

Invalid and indifference-band pixels never enter classification. Regression
keeps valid pixels because absolute error is continuous. Primary values are
`q=0.50`, `epsilon=0.50 px`, `band=0.10 px`; evaluation uses epsilon
0.25/0.50/1.00/3.00 and coverage 0.05/0.25/0.50/0.90.

## Universal evidence contract

All maps are `[B,1,144,180]`. Inputs are normalized raw disparity, raw x/y
gradients, local variance, raw-valid mask, BiDA-aligned t-1 disparity,
signed/absolute disagreement, aligned validity, warp support, FB error and
confidence, photometric residual, flow magnitude, and frozen A2 update
magnitude/g_error/c_memory. RGB is used only inside frozen SEA-RAFT and never
enters the detector. There is no backbone identity, confidence, stereo feature,
cost volume, hidden state, long memory or future frame.

## Detector and losses

The controlled ladder is pixel-wise 1x1, local 3x3 CNN, small two-scale CNN,
and the best encoder with positive error/uncertainty heads. Outputs are
`sigmoid(logit_error)`, `softplus(raw_mu)`, and
`softplus(raw_sigma)+1e-3`.

Loss ladder: classification; regression; joint; joint+Laplace uncertainty;
then indifference plus asymmetric clean-negative cost. Weighted BCE balances
positive/negative totals before applying the explicit false-positive ratios
1:1, 3:1 and 5:1. Clean authorization probability is separately penalized.

## Abstention contract

Authorization requires every condition:

```
p_error >= p_min
mu_error >= mu_min
sigma_error <= sigma_max
warp_support & aligned_valid
isfinite(u_A2) & abs(u_A2) <= 3 px
```

The final output is `d_raw + authorization * u_A2`; a rejected pixel is
bit-exact raw. Temperature and thresholds can be fitted only with an explicit
`split="validation"` guard.

Three modes are frozen on calibration sequences: ultra-safe (false update
<=10%, clean degradation <=5%), balanced (<=15%, <=10%), and high-coverage
diagnostic. Test sequences never influence thresholds.

## Promotion gate

A seen GO requires a useful calibrated low-coverage operating point, at least
two improving seen backbones, no severe collapse, false update <=15%, clean
degradation <=10%, and nontrivial retained A2 gain. Strong GO additionally
requires false update <10%, clean degradation <=5--10%, 30--50% retained A2
gain, and improvement on all three. Near-zero intervention is not a GO.

Only a seen GO authorizes the one-shot Fast-FoundationStereo test. Otherwise
the task stops without unseen access or joint fine-tuning.
