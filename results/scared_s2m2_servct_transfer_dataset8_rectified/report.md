# S2M2-S SERV-CT Fine-Tune Transfer To SCARED

## Setup

- Source model: S2M2-S pretrained.
- Fine-tuned model: S2M2-S SERV-CT honest holdout checkpoint.
- Fine-tuned checkpoint: `results/servct_s2m2_honest_finetune_gpu/finetune_refiners_250/s2m2_servct_finetuned.pth`
- SCARED subset: `dataset/scared_keyframes_gt_dataset8/dataset_8/keyframe_0..4`
- SCARED GT: `left_depth_map.tiff` point-cloud Z, converted to disparity with rectified `fx * baseline / Z`.
- Rectification: enabled using `endoscope_calibration.yaml` for left/right images and left depth map.
- GPU: CUDA via PyTorch.

## Why Rectification Matters

An initial unrectified evaluation produced meaningless errors: S2M2 predicted disparity at a different geometry/scale than the GT derived from the SCARED point-cloud TIFFs. The valid result is the rectified run in this folder.

## Results

| Run | Frames | Disp MAE px | Disp RMSE px | Bad-2 px % | Depth MAE mm | Depth RMSE mm | Depth Bad-2mm % |
|---|---:|---:|---:|---:|---:|---:|---:|
| pretrained S2M2-S | 5 | 4.2431 | 13.0358 | 17.0790 | 2.8372 | 7.3499 | 21.5904 |
| SERV-CT fine-tuned S2M2-S | 5 | 4.2832 | 12.8880 | 19.4818 | 2.8320 | 7.1476 | 22.6959 |

## First Read

The SERV-CT fine-tuned checkpoint does not clearly transfer to SCARED:

- disparity MAE is slightly worse: `4.2431 px` -> `4.2832 px`;
- disparity RMSE improves slightly: `13.0358 px` -> `12.8880 px`;
- depth MAE is almost unchanged/slightly better: `2.8372 mm` -> `2.8320 mm`;
- depth RMSE improves slightly: `7.3499 mm` -> `7.1476 mm`;
- Bad-1/Bad-2 rates worsen, suggesting more small/medium pixel errors even if some large outliers shrink.

This means the SERV-CT refiners-only tune is promising inside SERV-CT, but not yet a robust cross-dataset surgical adaptation. The next fine-tune should include SCARED GT or use a gentler adaptation objective.

## Artifacts

- Metrics: `metrics.csv`
- Summary: `summary.json`
- Montage: `scared_s2m2_base_vs_servct_finetuned_montage.png`
- Script: `scripts/scared/eval_scared_s2m2_transfer.py`

## Next Steps

1. Fine-tune with mixed SERV-CT + SCARED keyframe GT, holding out one SCARED dataset.
2. Try lower LR all-trainable adaptation and compare against refiners-only.
3. Add per-frame SCARED summaries to identify which keyframes benefit or degrade.
4. Extend from keyframes to warped SCARED video frames once the depth warp format is converted cleanly.
