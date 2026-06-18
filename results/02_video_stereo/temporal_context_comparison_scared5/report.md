# SCARED 5-Frame Temporal Context Comparison

Models: Stereo Any Video, S2M2-S frame-by-frame, Fast-FoundationStereo frame-by-frame.

## Status

This run is now considered a smoke-test comparison, not a valid temporal-context benchmark. The 5 frames are keyframes from different moments/scenes, not truly consecutive video frames. That makes it unsurprising that Stereo Any Video did not show a temporal advantage here.

## Question

Does a video-native stereo model show an immediate temporal-context advantage over strong frame-based stereo baselines on a short surgical stereo sequence?

## Setup

- Input: 5 rectified SCARED stereo keyframes from `results/video_stereo_repo_scouting/smoke_inputs/scared_rect5/`.
- Stereo Any Video: loaded the local `StereoAnyVideo_MIX.pth` checkpoint and inferred the sequence jointly.
- S2M2-S: inferred each stereo pair independently with the local S2M2 environment.
- Fast-FoundationStereo: reused existing frame-by-frame outputs on the same keyframes and converted depth to an inverse-depth proxy for visual/temporal comparison.
- No ground truth was used in this quick temporal pass.

Metric note: no optical-flow compensation or GT is used here. Adjacent delta is a quick flicker proxy on per-frame robust-normalized maps, so lower generally means smoother but can also mean over-smoothing.

| model | adjacent_delta_mean | adjacent_delta_p95_mean | frame_mean_std | frame_p95_std |
|---|---:|---:|---:|---:|
| stereoanyvideo | 0.288390 | 0.677411 | 0.123436 | 0.065167 |
| s2m2 | 0.272193 | 0.629787 | 0.104109 | 0.042205 |
| fast_foundation | 0.282449 | 0.659651 | 0.116403 | 0.058970 |

## First Read

On this 5-frame keyframe sequence, Stereo Any Video does not show a clear temporal-stability win yet. S2M2-S has the lowest quick flicker proxy, Fast-FoundationStereo is close, and Stereo Any Video is slightly higher.

Qualitatively the montage shows all three models producing broadly similar surgical structure. This is still encouraging as a runtime check: the video-native model runs on our data. For temporal conclusions, use `results/temporal_context_comparison_scared_consecutive32/`.

## Artifacts

- Visual comparison: `temporal_context_montage.png`
- Metrics CSV: `temporal_metrics.csv`
- Metrics JSON: `temporal_metrics.json`
- S2M2 frame outputs: `s2m2_disp_000.npy` to `s2m2_disp_004.npy`
- Run log: `run.log`

Repo-local copies were also placed under:

- `stereoanyvideo/results/temporal_context_scared5/`
- `s2m2/results/temporal_context_scared5/`
- `Fast-FoundationStereo/results/temporal_context_scared5/`

## Next Step

Repeat this comparison on a true consecutive SCARED clip, then compute flow-warped temporal consistency and GT disparity/depth error where ground truth is available. That will tell us whether the temporal model helps around specularities, tool motion, tissue deformation, and wound edges instead of only looking smooth.
