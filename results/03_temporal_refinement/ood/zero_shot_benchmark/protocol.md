# Zero-shot OOD protocol (frozen)

Goal: measure whether ARGOS refiners generalise, with **no** out-of-distribution tuning.

## Upstream stereo (raw input)

- **S2M2-S @ 512, pretrained** (`CH128NTR1.pth`), the exact backbone the refiners were
  trained to correct. Generated with the *unchanged* in-domain generator
  `data_prep/predict_s2m2_long_sequences.py` (`--variant S --width 512`): resize to width
  512 (isotropic), pad /32, fp16, crop, upsample disparity back to original resolution and
  divide by the scale factor → positive-px disparity in original image coordinates.
- No OOD finetuning. The SERV-CT-finetuned S2M2 checkpoint that exists in the repo is
  **not** used (it would contaminate the zero-shot upstream).

## Feature construction (identical to training)

- Reuses `train_tiny_refiner_v3_1_staged_abstention.make_features_from_raws` — the exact
  16-channel tensor: 4 causal disparity frames + 4 valid masks + {dt1, mean, median, var,
  |raw−median|, gx, gy, edge}, all ÷ `DISP_SCALE = 64`.
- **Causal window** `indices = [t, t−1, t−2, t−3]`, clamped at sequence start → **no future
  frame leakage**.
- **Grid**: `target_scale = 0.25`, `min_valid_ratio = 0.25` (identical to
  `s2m2_gt_refiner_targets_full`). SERV-CT 720×576 → 180×144, disparity kept in native px
  (no rescale to SCARED — that would be OOD tuning).
- **Valid mask** = `servct_valid & isfinite(gt) & (gt>0) & isfinite(raw)` then valid-aware
  area-average downsample — the same semantics used to build the training targets.
- OOD data is written as **training-format shards** (`raw_disp, gt_disp, valid_mask,
  delta_disp_gt_minus_raw`) so every model's own `FullFrameDataset` consumes it unchanged.

## Model policy (fairness)

Each model uses its **already-selected primary-dataset checkpoint, threshold, and proposal
scale**. No per-dataset checkpoint selection. See `servct/checkpoint_manifest.json`.

| model | checkpoint | residual scale | application |
|-------|-----------|---------------:|-------------|
| v3.2c | `tiny_refiner_v3_2c…_long/best.pt` | 3.0 | `refined = raw + (p_bad ≥ 0.7)·residual` (stored threshold) |
| EGBM-v1 | `experimental_refiner_vx_training/best.pt` | 3.0 | `refined = raw + residual` (internally gated) |
| EGBM-v2 | `egbm_v2_experimental/best.pt` | 3.0 | internally gated |
| EGBM-v2-CARE | `egbm_v2_care/best.pt` | 3.0 | internally gated |
| EGBM-v3-CARE-S | `egbm_v3_care_streaming/best.pt` | 3.0 | window (causal-replay); streaming degenerate on sparse OOD |
| MPC | `magnitude_proposal_critic_refiner/best_pareto.pt` | 32.0 | internally gated (large-proposal head) |
| CPV | `counterfactual_proposal_verifier_refiner/best_pareto.pt` | 32.0 | internally gated (proposal + verifier) |

Adding **Agent A's** final safe-fraction model = one `ModelEntry` appended in
`model_registry.py`; the harness and protocol are unchanged.

## Metrics

Computed on valid GT pixels at the 144×180 grid (same as in-domain lowres eval):
accuracy (MAE, RMSE, Bad-1/3/5, ΔMAE, relative improvement); correction accounting
(modified-pixel ratio, beneficial/harmful/neutral rates, net-benefit, mean beneficial
reduction, mean harmful increase, correction sign accuracy, magnitude ratio,
overshoot/undershoot); **safety** (new-Bad3/new-Bad1 from raw-good, catastrophic harmful
>6px, correction-magnitude percentiles + fraction >{3,6,12,20}px, top-{0.1,1,5,10}% damage
concentration); boundary-vs-interior; runtime. Per frame → per sequence → per model.

Because **no selected-oracle target exists OOD**, no oracle-gap recovery is reported. The
improvement metric is the plain `raw MAE − refined MAE` and its normalised form.
