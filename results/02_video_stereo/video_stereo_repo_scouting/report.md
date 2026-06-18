# ARGOS Video Stereo Repository Scouting

Date: 2026-06-08

Scope: scouting only. No training was started, no global Python packages were installed, and existing S2M2/Fast-FoundationStereo scripts were not modified.

Smoke input prepared:

`/home/pampaj/Desktop/stereo/results/video_stereo_repo_scouting/smoke_inputs/scared_rect5/`

This contains symlinks to 5 consecutive rectified SCARED pairs from:

`/home/pampaj/Desktop/stereo/Fast-FoundationStereo/data/surgical_stereo/scared/test_dataset_8_rectified/keyframe_{0..4}/`

Local smoke environment:

- `python3` is available.
- Base environment does not include `torch`, `cv2`, or `matplotlib`, so model execution was not attempted.
- Logs are in `/home/pampaj/Desktop/stereo/results/video_stereo_repo_scouting/logs/`.

## Summary Table

| model_name | repo_url | commit_hash | checkpoint_available | demo_runs | accepts_custom_stereo_sequence | input_format | output_format | temporal_window_or_memory | expected_resolution | requires_camera_pose_or_flow | estimated_integration_difficulty | recommended_next_action | notes |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| Stereo Any Video | https://github.com/TomTomTommi/stereoanyvideo | 3a8beddc470cfb7bf07dbc2bf4f04fc95d35d84c | Yes, README points to Google Drive checkpoints; default `StereoAnyVideo_MIX.pth` | No, blocked by missing base deps; `demo.py --help` fails at `cv2` import | Yes | Folder with `left/*.png|jpg` and `right/*.png|jpg`; default names in README are `left000000.png`, `right000000.png` but code sorts any png/jpg | `disparity.mp4`, `disparity_norm.mp4`, optional `disparity_norm_###.png`, `disparity_ori_###.png` | `--frame_size` default 150; `--kernel_size` default 20; model uses video batch tensor `[T,2,C,H,W]` | README/demo default resize `(720,1280)` | No pose/intrinsics for disparity demo; metric depth would still need focal/baseline outside repo | Medium | Integrate first. Build isolated conda env, download checkpoint, run 5-frame SCARED smoke, then add adapter that writes disparity PNG/NPY and computes temporal metrics | Highest fit for ARGOS: video-native, custom stereo folder input, temporal consistency. Apache-2.0. Clone checkout was partial due slow GitHub; selected files were materialized successfully. |
| TC-Stereo | https://github.com/jiaxiZeng/Temporally-Consistent-Stereo-Matching | ad714ad676265d9ed15a8fcd77a3cb35e0e3025f | Yes, README points to Dropbox checkpoints for TartanAir, SceneFlow, KITTI_raw | No, tarball clone failed after slow HTTP/2 stream; no local deps installed | Partially, likely with adapter | Dataset loaders expect TartanAir/SceneFlow/KITTI-like sequence roots with left/right images plus pose/calibration fields | Evaluation saves disparity PNG for KITTI and metrics; code has visualization utilities | Carries previous disparity, feature map, GRU/net state, previous pose, baseline/K params | Pads to divisibility by 32; KITTI/TartanAir examples use native sizes; max valid disparity often `<192` | Yes for temporal mode: K, baseline, current/previous pose transform; can disable temporal params only after first frame, but full method expects geometry | High | Integrate second if we can provide/endoscope-calibrate poses or approximate identity motion; otherwise use as geometry-aware upper baseline on datasets with pose | Strong scientific baseline, MIT license, PyTorch 2.0.1/CUDA 11.7, cupy_cuda117. More work than SAV because SCARED loader must provide K/baseline/pose. |
| TemporalStereo | https://github.com/youmi-zym/TemporalStereo | 0b5a94eb8fccce068a3cba8473190759cba42dc8 | Yes, README points to Google Drive checkpoints | No, `video_inference.py --help` fails at missing `matplotlib` in base env | Yes, but adapter required | `video_inference.py` expects `data_root/left`, `data_root/right`, `pose_left.txt`, optional `disp_gt`, with TartanAir-style pose parsing | `disp_0/*.png`, `color_disp/*.png`, `cats/*.png`, `error.txt` | One-step recurrent/previous-frame state via `last_outputs`; frame indices configurable in demo pipeline | Defaults `--resize-to-shape 384 1280`; README cites KITTI/TartanAir/SceneFlow configs | Yes: pose file, normalized K, baseline are used in video inference | High | Keep as third/practical fallback if TC-Stereo integration is blocked by pose handling or checkpoint access | Apache-2.0, Python 3.8, PyTorch 1.10.1+/CUDA 11.3, Apex, Detectron2, CuPy. Older and dependency-heavy. |
| PPMStereo | https://github.com/cocowy1/PPMStereo | d0ccf7705145502c1eea49e7be0ddeafbcfd6a08 | Yes, README points to Google Drive checkpoints | No, clone/tarball not completed; inspected raw README/tree/requirements/demo | Yes | Demo accepts `left/*.png|jpg` and `right/*.png|jpg` under `--path`, same style as Stereo Any Video; evaluation configs for Dynamic Replica/Sintel/real | `disparity.mp4`, `disparity_norm.mp4`, `left_right.mp4`, optional disparity PNGs | Pick-and-Play Memory, compact long-range dynamic buffer; demo exposes `--frame_size`, `--kernel_size`, `--iters` | README demo default `(720,1280)`; training examples use `image_size 320 512`; Dynamic Replica high-res is heavy | No for folder demo disparity; evaluation/training may require dataset-specific metadata | Medium-high | Scout after SAV/TC; use only if checkpoint download is straightforward and GPU memory is sufficient | MIT license. Very relevant and recent, but README recommends >=48GB GPU for Dynamic Replica evaluation. Good exploratory baseline, not first integration. |
| TemporallyConsistentDepth | https://github.com/facebookresearch/TemporallyConsistentDepth | be85390cf5db72a996bebba3d9f34439f1576196 | Yes, repo includes/listed small fusion weights and `initialize.sh` downloads DPT/RAFT-Stereo backbones | No, optional repo not locally cloned; raw README/tree inspected | Not directly; custom data loader required | Dataset loader must return RGB, unprocessed/scaled depth, camera pose, intrinsics; stereo only through RAFT-Stereo wrapper for supported datasets | Visual comparison RGB/InputDepth/Result; optional floating-point depth `.npy` via `--save_numpy` | Online point-based fusion over video | Not specified as fixed, but tested on ScanNet/COLMAP/MPI-Sintel style data | Yes: camera pose and intrinsics are required; disparity must be converted to scaled depth first | High | Do not integrate as stereo baseline now; reserve as temporal post-processing/depth-fusion baseline after we have calibrated SCARED pose/intrinsics | CC-BY-NC 4.0, non-commercial. Useful idea, but not pure stereo matching and complicates metric depth handling. |

## Smoke Test Commands Tried

```bash
python3 demo.py --help
```

Run in `stereoanyvideo`; result: failed before argparse with `ModuleNotFoundError: No module named 'cv2'`.

```bash
python3 projects/TemporalStereo/video_inference.py --help
```

Run in `TemporalStereo`; result: failed before argparse with `ModuleNotFoundError: No module named 'matplotlib'`.

No smoke test was attempted for TC-Stereo, PPMStereo, or TemporallyConsistentDepth because full local materialization was incomplete or optional, and installing dependencies globally is out of scope.

## Clone/Materialization Status

- `stereoanyvideo`: partial Git clone at the target commit. README/LICENSE and selected inference files are present.
- `TemporalStereo`: shallow clone is present and usable for inspection.
- `Temporally-Consistent-Stereo-Matching`: commit hash recorded; tarball download failed after a slow HTTP/2 stream closed early.
- `PPMStereo`: commit hash recorded; raw tree/README/requirements/demo inspected; tarball loop was stopped after TC failure and slow transfer.
- `TemporallyConsistentDepth`: commit hash recorded; raw tree/README/environment/license inspected.

## Recommended First Integrations

1. Stereo Any Video.

It is the best first ARGOS temporal baseline because it accepts a simple rectified stereo image-folder sequence, does not require camera pose for disparity inference, has public checkpoints, outputs video/PNG disparities, and directly targets temporally consistent stereo video.

2. TC-Stereo.

It is the strongest second scientific baseline if we can provide SCARED camera intrinsics, baseline, and a reasonable pose stream. It is more invasive than SAV, but valuable because it explicitly carries temporal geometry/state and is an ECCV 2024 method. If pose handling blocks progress, use TemporalStereo as the fallback second integration and keep TC-Stereo for calibrated sequences only.
