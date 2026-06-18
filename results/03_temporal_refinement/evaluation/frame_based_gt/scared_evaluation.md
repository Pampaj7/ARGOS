# SCARED Evaluation

Single GT-backed SCARED table. The main table below only includes
methods evaluated through the `scared_temporal_gt_dataset9_keyframe3` protocol in
`scripts/scared/run_all_scared_baselines.py`.

| Method | Training / Checkpoint | Input res. | Depth MAE ↓ | Bad-2 mm ↓ | Disp. MAE ↓ | Runtime ↓ | VRAM ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S2M2-S | CH128NTR1.pth | 512 width | 2.513 | 37.49% | 8.493 | 57.8 ms | 0.36 GB |
| S2M2-L | CH256NTR3.pth | 736 width | 2.532 | 37.79% | 8.466 | 178.9 ms | 1.63 GB |
| Fast-FoundationStereo ONNX | 320x736 | 320x736 | 2.532 | 37.53% | 8.421 | 44.1 ms |  |
| S2M2-L full | CH256NTR3.pth | 1024x1280 full | 2.534 | 37.81% | 8.457 | 491.4 ms | 2.83 GB |
| S2M2-XL | CH384NTR3.pth | 1024x1280 full | 2.540 | 37.84% | 8.432 | 894.6 ms | 5.05 GB |
| SGBM | OpenCV SGBM block=3 max_disp=320 | 1024x1280 full | 2.700 | 37.83% | 7.898 | 94.9 ms |  |

## Protocol

- Dataset: `dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3/metadata.csv`.
- Frames: 130 samples with left/right images, calibration, and depth/disparity GT.
- Images are rectified inside the shared loader for keyframes, or loaded from pre-rectified warped metadata for warped samples.
- Resized predictions are rescaled back to original disparity coordinates.
- Metrics: disparity MAE and metric-depth MAE/Bad-2 mm over the shared valid GT mask.
- Important: this is a same-evaluator / same-GT table, not a same-input-resolution table. The `Input res.` column is part of the result.
- Fast-FoundationStereo currently uses the only local Fast-FoundationStereo ONNX artifact available: fixed `320x736`.
- S2M2-S/L rows intentionally use deployment-style widths (`512`, `736`, or full), while external models currently run on the native rectified resolution unless otherwise stated.

## SERV-CT Methods Not Yet In This SCARED Table

| Method | Status | Notes |
| --- | --- | --- |
| StereoAnyVideo | separate_video_adapter | Use `scripts/scared/run_stereoanyvideo_temporal_eval.py`; not frame-adapter compatible yet. |

## Evidence

- Per-method outputs: `results/temporal evaluation/frame_based_gt/native_frame_methods`.
- Row evidence: `evidence.csv`.
