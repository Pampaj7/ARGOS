# Temporal Evaluation With SCARED GT

Sequence: `/dtu/p1/leopam/ARGOS/dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3`

This table is regenerated on rectified frames with GT attached. The old long temporal cache is not scored against this GT because its frames are not pixel-aligned with the newly converted rectified SCARED frames.

| method | training_or_checkpoint | input_res | frames_with_gt_used | Depth MAE ↓ | Bad-2 mm ↓ | Disp. MAE ↓ | Temporal diff ↓ | Runtime ↓ | VRAM ↓ | causal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TCSM |  |  | 103 | 2.958 | 38.022 | 8.919 | 2.158 | 3151.871 |  |  |
| PPMStereo |  |  | 103 | 2.864 | 38.839 | 9.065 | 0.991 | 1035.549 |  |  |
| RAFT-Stereo | /dtu/p1/leopam/ARGOS/external/frame_stereo_repos/RAFT-Stereo/models/raftstereo-sceneflow.pth |  | 103 | 2.629 | 37.189 | 8.379 | 1.034 |  |  |  |
| DEFOM-Stereo | /dtu/p1/leopam/ARGOS/external/frame_stereo_repos/DEFOM-Stereo/checkpoints/defomstereo_vitl_eth3d.pth |  | 103 | 2.585 | 36.838 | 8.272 | 0.992 |  |  |  |
| MonSter++ | /dtu/p1/leopam/ARGOS/external/frame_stereo_repos/MonSter-plusplus/MonSter++/checkpoints/Mix_all_large.pth |  | 103 | 2.627 | 37.231 | 8.333 | 1.012 |  |  |  |
| StereoAnywhere |  |  | 0 |  |  |  |  | 14636.185 |  | No |
| S2M2-S@512 | S | 512 | 103 | 2.584 | 37.089 | 8.384 | 0.988 |  | 0.000 |  |
| S2M2-L@736 | L | 736 | 103 | 2.593 | 37.144 | 8.331 | 0.984 |  | 0.000 |  |
| StereoAnyVideo@384x640 |  | 384x640 | 103 | 2.588 | 36.694 | 8.251 | 0.925 |  | 0.000 |  |

## Protocol

- GT source: raw `dataset_9.zip`, block `keyframe_3`, converted into rectified left/right plus depth/disparity/mask.
- Geometry metrics use only frames whose GT valid-pixel ratio passes the configured threshold.
- Disparity predictions are saved in original image coordinates before metric computation.
- StereoAnyVideo is run in chunks; it remains a video-native teacher/baseline, but chunking can slightly affect long-range temporal context.
- Temporal smoothness is not geometric correctness; here we report both GT errors and temporal differences.

## Methods Included

- `TCSM`
- `PPMStereo`
- `RAFT-Stereo`
- `DEFOM-Stereo`
- `MonSter++`
- `StereoAnywhere`
- `S2M2-S@512`
- `S2M2-L@736`
- `StereoAnyVideo@384x640`
