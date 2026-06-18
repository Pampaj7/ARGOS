# SCARED 32-Frame Temporal Context Comparison

Models: Stereo Any Video, S2M2-S frame-by-frame, Fast-FoundationStereo ONNX frame-by-frame.

## Runtime

All three model runs are GPU-backed in this final pass:

- Stereo Any Video: CUDA via PyTorch.
- S2M2-S: CUDA via PyTorch.
- Fast-FoundationStereo: ONNX Runtime with active providers `CUDAExecutionProvider`, `CPUExecutionProvider`.

Fast-FoundationStereo previously fell back to CPU because ONNX Runtime could not see CUDA libraries from the conda env. The rerun fixed this by exporting all `nvidia/*/lib` paths in `LD_LIBRARY_PATH`.

Fast-FoundationStereo GPU summary:

`Fast-FoundationStereo/results/argos_scared_consecutive32_gpu/summary.json`

## Why This Replaces The 5-Frame Test

The earlier `temporal_context_comparison_scared5` run used SCARED keyframes, not truly consecutive frames. That was useful for smoke testing, but it was not a fair way to test temporal context.

This run uses 32 consecutive frames extracted from `test_dataset_9/keyframe_3/rgb.mp4`. The video is top/bottom stereo at 1280x2048, split into:

- top half: left image;
- bottom half: right image.

Input sequence:

`argos_data/scared_consecutive/test_dataset_9_keyframe_3_32/`

## Metrics

No optical-flow compensation or GT is used here. Adjacent delta is a quick temporal-error proxy on per-frame robust-normalized maps, so lower generally means smoother but can also mean over-smoothing.

| model | adjacent_delta_mean | adjacent_delta_p95_mean | frame_mean_std | frame_p95_std |
|---|---:|---:|---:|---:|
| stereoanyvideo | 0.010267 | 0.026622 | 0.019602 | 0.099152 |
| s2m2 | 0.013799 | 0.033135 | 0.024044 | 0.096444 |
| fast_foundation | 0.025226 | 0.111651 | 0.029540 | 0.014414 |

## Error Maps

The comparison includes temporal error maps in addition to depth/disparity maps.

Current error map definition:

`error_t = abs(robust_norm(pred_t) - robust_norm(pred_{t-1}))`

Frame 0 has no previous frame, so its temporal error map is zero/black. These are not GT error maps yet. Once we have aligned GT depth/disparity for a sequence, we should add geometric error maps too.

## First Read

On this corrected consecutive-frame test, Stereo Any Video shows a real temporal-stability advantage over the frame-by-frame baselines:

- compared with S2M2-S, adjacent delta mean improves from `0.013799` to `0.010267`;
- compared with S2M2-S, adjacent delta p95 improves from `0.033135` to `0.026622`;
- compared with Fast-FoundationStereo ONNX, adjacent delta mean improves from `0.025226` to `0.010267`;
- visually, the Stereo Any Video temporal error maps are cleaner;
- S2M2 has more small local temporal spikes around bright/specular regions;
- Fast-FoundationStereo shows a stronger unstable/saturated band near the right side in this fixed-resolution ONNX run.

This is the expected direction for a video-native stereo model. It does not prove better depth accuracy yet, because this metric has no ground truth and no flow warping. It does support the hypothesis that temporal context is useful on surgical stereo video.

## Artifacts

- Depth-only montage: `temporal_context_montage.png`
- Depth plus temporal-error montage: `temporal_context_depth_error_montage.png`
- Per-frame temporal error maps: `temporal_error_maps/<model>/temporal_error_###.png`
- Temporal error stacks: `temporal_error_maps/<model>/temporal_error.npy`
- Metrics CSV: `temporal_metrics.csv`
- Metrics JSON: `temporal_metrics.json`
- Stereo Any Video outputs: `results/stereoanyvideo_scared_consecutive32/images/`
- S2M2 per-frame outputs: `s2m2_disp_000.npy` to `s2m2_disp_031.npy`
- Fast-FoundationStereo ONNX GPU outputs: `Fast-FoundationStereo/results/argos_scared_consecutive32_gpu/`

## Next Step

Compute a flow-warped temporal error. After that, repeat on a sequence with available GT depth/disparity so temporal stability can be checked against actual geometric accuracy.
