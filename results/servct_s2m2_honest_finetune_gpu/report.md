# S2M2-S SERV-CT Honest Fine-Tuning

## Setup

- Model: S2M2-S.
- Data source: `ARGOS/dataset/raw/surgical_stereo/servct/SERV-CT`.
- Split: honest holdout.
- Train: `Experiment_1`, `Reference_CT`.
- Test: `Experiment_2`, `Reference_CT`.
- Fine-tune mode: refiners only.
- Steps: 250.
- Learning rate: `2e-5`.
- GPU: CUDA via PyTorch.

## Results On Experiment_2

| Run | Frames | Disp MAE px | Disp RMSE px | Bad-2 px % | Depth MAE mm | Depth RMSE mm | Depth Bad-2mm % |
|---|---:|---:|---:|---:|---:|---:|---:|
| pretrained S2M2-S | 8 | 1.4615 | 3.3191 | 14.2851 | 1.7638 | 3.6088 | 23.0564 |
| fine-tuned refiners 250 | 8 | 1.4684 | 2.7885 | 16.1559 | 1.4580 | 2.7447 | 14.4192 |

## First Read

This run improves metric depth while leaving disparity MAE essentially flat/slightly worse:

- depth MAE improves by `0.3058 mm`;
- depth RMSE improves by `0.8641 mm`;
- depth Bad-2mm improves by `8.64 percentage points`;
- disparity MAE changes from `1.4615 px` to `1.4684 px`, so it does not improve on average;
- disparity RMSE improves, suggesting fewer large disparity outliers even though the mean absolute disparity error is not better.

This is a useful but mixed result. The fine-tune is improving metric depth behavior on the holdout, but the disparity objective is not uniformly better. The next run should tune loss weighting and/or train more of the model, then compare against this as the honest baseline.

## Figures

- Before/after error montage: `s2m2_servct_before_after_error_montage.png`
- Pretrained eval montage: `baseline_pretrained_eval/montage_left_pred_gt_err.png`
- Fine-tuned eval montage: `finetune_refiners_250/eval/montage_left_pred_gt_err.png`

Note: the historical montage error maps use per-image display scaling. Use the numeric metrics for exact comparisons.

## Artifacts

- Baseline metrics: `baseline_pretrained_eval/metrics.csv`
- Baseline summary: `baseline_pretrained_eval/summary.json`
- Fine-tuned metrics: `finetune_refiners_250/eval/metrics.csv`
- Fine-tuned summary: `finetune_refiners_250/eval/summary.json`
- Comparison summary: `comparison_summary.json`
- Fine-tuned checkpoint: `finetune_refiners_250/s2m2_servct_finetuned.pth`
- Training log: `finetune_refiners_250/train_log.json`

## Recommended Next Runs

1. Run `S2M2-S all-trainable` with a lower LR, e.g. `5e-6`, to see if disparity MAE improves without destroying depth.
2. Run a shorter low-loss-weight variant with less gradient/confidence auxiliary loss.
3. Repeat with S2M2-M if memory allows.
4. Add fixed-scale GT error maps so visual before/after colors are directly comparable across runs.
