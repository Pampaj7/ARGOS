# ARGOS Model Zoo

This document tracks the model baselines relevant for ARGOS.

## Tested

| Model | Local Path | Checkpoint | Status |
|---|---|---|---|
| S2M2-S/M | `../external/frame_stereo_repos/s2m2` | HF pretrained S/M | tested on SERV-CT |
| S2M2-S fine-tuned | `../external/frame_stereo_repos/s2m2` | SERV-CT adapted | tested on SERV-CT |
| Fast-FoundationStereo ONNX | `../external/frame_stereo_repos/Fast-FoundationStereo` | ONNX demo weights | tested on SERV-CT |
| Stereo Anywhere | `../external/frame_stereo_repos/stereoanywhere` | SceneFlow + DepthAnything V2 | tested on SERV-CT |
| MonSter++ large | `../external/frame_stereo_repos/MonSter-plusplus` | `Mix_all_large.pth` | tested on SERV-CT |
| RT-MonSter++ | `../external/frame_stereo_repos/MonSter-plusplus` | `Zero_shot.pth` | tested on SERV-CT |
| RAFT-Stereo | `../external/frame_stereo_repos/RAFT-Stereo` | RVC/Middlebury | tested on SERV-CT |
| CREStereo | `../external/frame_stereo_repos/stereo_matching_crestereo` | bundled `epoch-570.pth` | tested on SERV-CT |
| StereoAnyVideo | `../external/video_stereo_repos/stereoanyvideo` | Temporal stereo | Video baseline |SERV-CT |
| OpenCV SGBM | local script | none | tested on SERV-CT |

## Prepared Or Pending

| Model | Local Path | Blocker | Why It Matters |
|---|---|---|---|
| S2M2-L/XL | `../external/frame_stereo_repos/s2m2` | queued weights | scaling test for best current family |
| DEFOM-Stereo | `../external/frame_stereo_repos/DEFOM-Stereo` | VIT-L Middlebury best so far; VIT-L SceneFlow unstable | depth-foundation stereo baseline |
| IGEV++ | `../external/frame_stereo_repos/IGEV-plusplus` | Google Drive/manual weights | strong pure stereo baseline |
| Selective-Stereo | `../external/frame_stereo_repos/Selective-Stereo` | Google Drive/manual weights | detail/frequency baseline |
| FoundationStereo full | `../external/frame_stereo_repos/FoundationStereo` | Google Drive quota | full NVLabs foundation baseline |

## Current Interpretation

S2M2 is the strongest family so far. MonSter++ large is close enough to be a serious paper baseline, especially because monodepth priors may help with low-texture tissue. RAFT-Stereo and CREStereo are useful reviewer-proof baselines, but are weaker on current SERV-CT depth metrics.
