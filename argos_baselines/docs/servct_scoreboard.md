# ARGOS SERV-CT Baseline Scoreboard

Common benchmark: SERV-CT Reference_CT, 16 rectified stereo frames unless noted.
Metrics are disparity-space and metric-depth errors against GT disparity/depth.

## Scores

| Rank | Model | Disp MAE px | Disp RMSE px | Bad-2 % | Depth MAE mm | Depth RMSE mm | Frames | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | S2M2-S fine-tuned all-surgical | 0.810 | 1.843 | 5.48 | 1.022 | 2.309 | 16 | SERV-CT adapted checkpoint; not independent holdout |
| 2 | S2M2-S zero-shot | 1.281 | 2.730 | 13.18 | 1.711 | 3.386 | 16 | pretrained S checkpoint |
| 3 | S2M2-M zero-shot | 1.292 | 2.509 | 15.13 | 1.731 | 3.366 | 16 | pretrained M checkpoint |
| 4 | MonSter++ MixAll large i16 | 1.460 | 2.846 | 16.43 | 1.772 | 3.281 | 16 | large MixAll checkpoint, 16 iterations |
| 5 | Fast-FoundationStereo ONNX | 1.480 | 2.821 | 18.07 | 1.794 | 3.291 | 16 | NVLabs real-time foundation baseline |
| 6 | DEFOM-Stereo VIT-L Middlebury | 1.733 | 4.623 | 15.85 | 1.989 | 4.295 | 16 | VIT-L Middlebury checkpoint |
| 7 | RT-MonSter++ zero-shot | 1.601 | 2.785 | 21.55 | 2.048 | 3.425 | 16 | RT zero-shot checkpoint |
| 8 | Stereo Anywhere VIT-L | 1.696 | 3.026 | 20.50 | 2.053 | 3.912 | 16 | Depth Anything V2-L prior |
| 9 | DEFOM-Stereo VITS RVC | 1.767 | 4.427 | 17.24 | 2.072 | 4.548 | 16 | VITS RVC checkpoint |
| 10 | RAFT-Stereo RVC | 1.846 | 4.740 | 18.67 | 2.195 | 4.506 | 16 | iRAFT RVC checkpoint, context_norm=instance |
| 11 | CREStereo | 1.821 | 3.311 | 25.00 | 2.324 | 4.298 | 16 | bundled epoch-570 checkpoint |
| 12 | RAFT-Stereo Middlebury | 1.794 | 4.092 | 19.14 | 2.397 | 5.082 | 16 | Middlebury checkpoint |
| 13 | DEFOM-Stereo VITS SceneFlow | 2.412 | 7.726 | 22.22 | 2.410 | 5.343 | 16 | VITS SceneFlow checkpoint; smoke baseline |
| 14 | SGBM | 8.379 | 15.196 | 47.51 | 30.870 | 180.669 | 16 | classical OpenCV baseline |

## Repository Status

| Repo | Status | Why it matters |
|---|---|---|
| RAFT-Stereo | cloned and tested; Dropbox models downloaded | historical reviewer baseline |
| IGEV++ | cloned; weights are Google Drive gated/manual | strong pure stereo baseline |
| Selective-Stereo | cloned; weights are Google Drive gated/manual | CVPR 2024 Highlight, detail/frequency baseline |
| CREStereo | cloned and tested | practical robust stereo baseline |
| MonSter++ | cloned; RT and large tested | monodepth-prior foundation stereo |
| DEFOM-Stereo | cloned and tested; VIT-L Middlebury is best DEFOM run so far, VIT-L SceneFlow was unstable on SERV-CT | depth-foundation stereo baseline |

## Local Modifications

- `stereo_matching_crestereo/stereo_matching_crestereo/stereo_matching.py`: patched resize guard so `input_hw=None` does not trigger `boxx.resize` with NumPy 2.x.
- `stereo_matching_crestereo/scripts_eval_servct_crestereo.py`: added SERV-CT evaluator and montage writer.
- `MonSter-plusplus/MonSter++/core/monster.py` and `RT-MonSter++/core/monster.py`: patched DepthAnything checkpoint path to local `checkpoints/depth_anything_v2_{encoder}.pth`.
- `MonSter-plusplus/*/scripts_eval_servct_monster.py`: added SERV-CT evaluator for RT and large checkpoints.
- `DEFOM-Stereo/checkpoints/depth_anything_v2_{vits,vitl}.pth`: symlinked to existing Stereo Anywhere DepthAnything V2 weights.
- `DEFOM-Stereo/scripts_eval_servct_defom.py`: added SERV-CT evaluator and montage writer.
- `argos_baselines/scripts/make_servct_scoreboard.py`: creates this report and PNG ranking.

## Next Baselines

- Add S2M2-L/XL once the queued checkpoint download starts after SCARED.
- Add IGEV++ once weights are available from Google Drive or a usable mirror.
- Add Selective-Stereo once weights are available.
- Optionally test more DEFOM checkpoints/iteration settings after SCARED conversion.
