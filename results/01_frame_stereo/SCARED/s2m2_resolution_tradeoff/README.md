# S2M2 Resolution Tradeoff On SCARED

Resolution tradeoff benchmark for S2M2 on rectified SCARED dataset_8 keyframes.

Primary question: identify the best practical deployment candidate by accuracy, runtime, and VRAM.

## Dataset

`/home/pampaj/Desktop/ARGOS/dataset/scared_keyframes_gt_dataset8/dataset_8`

This is the same rectified 5-keyframe SCARED subset used by the previous size-tradeoff benchmark.

## Models And Resolutions

- S2M2-S: full, 1024, 736, 512
- S2M2-L: full, 1024, 736, 512
- S2M2-XL: retained as reference

Resized predictions are projected back to original disparity coordinates:

```python
pred_disp_original = pred_disp_resized / scale_x
```

## Main Outputs

- `s2m2_resolution_tradeoff.md`: practical analysis and Pareto frontier.
- `s2m2_resolution_tradeoff.csv`: aggregate metrics.
- `s2m2_resolution_tradeoff.json`: machine-readable metrics.
- `s2m2_size_tradeoff_frame_metrics.csv`: per-frame metrics from the shared benchmark runner.
- `qualitative/`: left image, GT disparity, GT depth, predictions, and absolute error maps.
- `run.log`: execution log.

The `s2m2_size_tradeoff.*` files in this folder are raw outputs from the shared benchmark runner; the official resolution-tradeoff report is `s2m2_resolution_tradeoff.*`.

## Current Takeaway

- Best S/L candidate under 500 ms: `L@full`, depth MAE `2.7100 mm`, runtime `485.10 ms`, VRAM `2899.5 MB`.
- Best S/L candidate under 300 ms: `L@736`, depth MAE `2.7425 mm`, runtime `185.86 ms`, VRAM `1671.1 MB`.
- Lowest VRAM S/L candidate: `S@512`, depth MAE `2.8660 mm`, runtime `69.55 ms`, VRAM `374.9 MB`.

Recommendation: use `L@736` as the practical deployment candidate, `L@full` as the best under-500-ms evaluation candidate, and `XL@full` only as a teacher/reference unless larger SCARED runs show bigger hard-frame gains.

