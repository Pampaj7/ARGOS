# SCARED Evaluation

Single GT-backed SCARED table. The main table below only includes
methods evaluated through the `scared_warped_gt_108` protocol in
`scripts/scared/run_all_scared_baselines.py`.

| Method | Training / Checkpoint | Input res. | Depth MAE ↓ | Bad-2 mm ↓ | Disp. MAE ↓ | Runtime ↓ | VRAM ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S2M2-S | CH128NTR1.pth | 512 width | 3.512 | 45.61% | 6.872 | 58.1 ms | 0.36 GB |
| RT-MonSter++ zero-shot | Zero_shot.pth | native warped_gt_108 | 3.543 | 45.66% | 6.449 | 129.1 ms | 2.40 GB |
| StereoAnywhere | stereoanywhere_sceneflow + DepthAnything-V2-S | native warped_gt_108 | 3.757 | 47.38% | 7.174 | 1504.0 ms | 8.18 GB |
| MonSter++ MixAll | Mix_all_large.pth | native warped_gt_108 | 4.112 | 46.92% | 11.771 | 1869.5 ms | 4.56 GB |
| S2M2-XL | CH384NTR3.pth | 1024x1280 full | 5.804 | 49.14% | 17.435 | 892.9 ms | 5.05 GB |
| Fast-FoundationStereo ONNX | 320x736 | 320x736 | 5.811 | 48.49% | 16.612 | 44.1 ms |  |
| RAFT-Stereo Middlebury | raftstereo-middlebury.pth | native warped_gt_108 | 5.829 | 50.45% | 15.107 | 903.7 ms | 3.34 GB |
| S2M2-L | CH256NTR3.pth | 736 width | 6.522 | 51.44% | 18.542 | 179.0 ms | 1.63 GB |
| S2M2-L full | CH256NTR3.pth | 1024x1280 full | 7.817 | 51.62% | 31.831 | 489.8 ms | 2.83 GB |
| DEFOM-Stereo ViT-L ETH3D | defomstereo_vitl_eth3d.pth | native warped_gt_108 | 8.460 | 50.22% | 38.547 | 1633.5 ms | 5.25 GB |
| CREStereo | local checkpoint | native warped_gt_108 | 8.641 | 51.46% | 29.833 | 436.8 ms | 1.26 GB |
| SGBM | OpenCV SGBM block=3 max_disp=320 | 1024x1280 full | 38.955 | 53.96% | 16.388 | 96.9 ms |  |

## Protocol

- Dataset: `dataset/SCARED/curated/warped_gt_108/metadata.csv`.
- Frames: 108 samples with left/right images, calibration, and depth/disparity GT.
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

- Per-method outputs: `/home/pampaj/Desktop/ARGOS/results/scared evaluation/warped_gt_108`.
- Row evidence: `evidence.csv`.
