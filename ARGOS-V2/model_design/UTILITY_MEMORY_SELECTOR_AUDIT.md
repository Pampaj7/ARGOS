# ARGOS v2 — Utility-Aware Causal Memory Selector Audit

## Scope

This experiment is deliberately narrower than the previous A2/refiner and
detector studies.  It trains one causal selector for the binary action
`raw_t` versus `warp(raw_{t-1}, flow_{t->t-1})`.  It creates no residual,
does not modify any frozen component, and has no future-frame access.

## Frozen BiDA contract

The canonical implementation is
`model_design/external_components/bidavideo.py::causal_warp` and
`::temporal_disparity_evidence`.  SEA-RAFT takes first image to second image.
For causal target/current-to-source/past sampling, its flow is `current->past`.
At a target pixel `x`, source is sampled at `x + flow(x)` with bilinear
`grid_sample`, zero padding, `align_corners=True`, and `(W-1,H-1)` coordinate
normalization.  It returns in-bounds support and a conservatively sampled
source-valid mask; neither disparity unit nor magnitude is scaled because both
maps use cache grid `144x180`.

The paired supervised mask is exactly:

```text
GT coverage > 0.50
& raw prediction valid
& sampled memory valid
& in-bounds warp support
```

FB confidence is evidence only, not a validity gate.  This is the mask used by
the large-scale BiDA audit, where the per-pixel oracle was defined as
`min(|raw-GT|, |aligned_memory-GT|)`.

## Signal motivating this selector

`results/large_scale_bida_signal_audit/aggregate_summary.json` reports a seen
pool raw EPE of 0.26659, direct memory EPE 0.26643, and raw-or-memory oracle
EPE 0.22101: a 0.04558 px / 17.10% available gain.  The oracle gain was
positive in all 51 seen backbone-sequence groups.  Memory wins on 49.79% and
loses on 50.14% of common pixels; therefore direct replacement is invalid but
the decision problem is non-trivial.  The unseen aggregate oracle gain was
0.02439 px / 10.50%, also positive in every backbone-sequence group.

## Target

At every common pixel:

```text
e_raw = |raw_t - GT_t|
e_mem = |aligned_memory_t - GT_t|
u = e_raw - e_mem
```

`u > 0` means memory improves geometry.  A predeclared cache-grid margin
epsilon=0.10 px avoids treating numerical ties as useful.  The model predicts
`P(u > epsilon)`, `E[max(u-epsilon, 0)]`, and
`E[max(-u-epsilon, 0)]`.  Its expected utility is the difference of the latter
two heads.  GT-derived target tensors never enter its evidence.

## Inputs and normalization

All maps are backbone-independent and on `[B,C,144,180]`:

1. raw disparity / 64;
2. aligned t-1 disparity / 64;
3. signed memory-minus-raw residual / 16;
4. absolute residual / 16;
5-6. flow x/y / 32;
7. flow magnitude / 32;
8. FB confidence;
9-11. warp support, aligned-valid and raw-valid masks;
12-13. raw and aligned-memory gradient magnitudes / 4.

All continuous values are clipped to the indicated range.  There is no RGB,
backbone name/identity, cost volume, stereo confidence, A2 proposal, detector,
or future frame.

## Architecture and loss

`models/utility_memory_selector.py::UtilityMemorySelector` is a 3x3
GroupNorm-SiLU CNN with four residual blocks and independent probability,
positive-gain and harmful-magnitude heads.  Default capacity is recorded from
the instantiated model and constrained to 100k–1M trainable parameters.

The fixed loss is weighted BCE for `u>epsilon`, smooth-L1 for positive gain and
harm magnitude, and an asymmetric expected harmful-selection penalty.  Harmful
pixels receive fourfold classification weight.  There is no temporal-smoothing
loss.

At inference raw is default.  Memory is accepted only if probability, expected
utility, expected harm and causal support pass a calibration threshold selected
exclusively on `dataset_7_keyframe_1/2`.  Rejection is `torch.where` to raw and
is bit-exact.

## Split and independence

Training: 13 accepted non-dataset-7 sequences, 12,852 t-1 pairs per backbone.
Validation/calibration: dataset_7 keyframes 1/2, 1,273 pairs/backbone.  Final
seen test: keyframes 3/4, 2,779 pairs/backbone.  Training uses only S2M2-S,
RAFT-Stereo and StereoAnywhere.  Fast-FoundationStereo and CREStereo are not
loaded until a seen checkpoint and operating point are frozen.

The balanced sampler equalizes each `(backbone, sequence)` group by repeating
only shorter groups; every original pair appears at least once per epoch and
groups are interleaved.  Statistical confidence intervals resample complete
sequences, not pixels.

Historical note: dataset_7 was used by earlier A2 work.  Within this selector
experiment, 3/4 are strictly excluded from training, model selection and
calibration; this limitation is disclosed in `split_audit.json`.
