# ARGOS Model Zoo

This document tracks the model baselines relevant for ARGOS.

## Tested

| Model | Local Path | Checkpoint | Status |
|---|---|---|---|
| S2M2-S/M | `/home/pampaj/Desktop/stereo/s2m2` | HF pretrained S/M | tested on SERV-CT |
| S2M2-S fine-tuned | `/home/pampaj/Desktop/stereo/s2m2` | SERV-CT adapted | tested on SERV-CT |
| Fast-FoundationStereo ONNX | `/home/pampaj/Desktop/stereo/Fast-FoundationStereo` | ONNX demo weights | tested on SERV-CT |
| Stereo Anywhere | `/home/pampaj/Desktop/stereo/stereoanywhere` | SceneFlow + DepthAnything V2 | tested on SERV-CT |
| MonSter++ large | `/home/pampaj/Desktop/stereo/MonSter-plusplus` | `Mix_all_large.pth` | tested on SERV-CT |
| RT-MonSter++ | `/home/pampaj/Desktop/stereo/MonSter-plusplus` | `Zero_shot.pth` | tested on SERV-CT |
| RAFT-Stereo | `/home/pampaj/Desktop/stereo/RAFT-Stereo` | RVC/Middlebury | tested on SERV-CT |
| CREStereo | `/home/pampaj/Desktop/stereo/stereo_matching_crestereo` | bundled `epoch-570.pth` | tested on SERV-CT |
| OpenCV SGBM | local script | none | tested on SERV-CT |

## Prepared Or Pending

| Model | Local Path | Blocker | Why It Matters |
|---|---|---|---|
| S2M2-L/XL | `/home/pampaj/Desktop/stereo/s2m2` | queued weights | scaling test for best current family |
| DEFOM-Stereo | `/home/pampaj/Desktop/stereo/DEFOM-Stereo` | VIT-L Middlebury best so far; VIT-L SceneFlow unstable | depth-foundation stereo baseline |
| IGEV++ | `/home/pampaj/Desktop/stereo/IGEV-plusplus` | Google Drive/manual weights | strong pure stereo baseline |
| Selective-Stereo | `/home/pampaj/Desktop/stereo/Selective-Stereo` | Google Drive/manual weights | detail/frequency baseline |
| FoundationStereo full | `/home/pampaj/Desktop/stereo/FoundationStereo` | Google Drive quota | full NVLabs foundation baseline |

## Current Interpretation

S2M2 is the strongest family so far. MonSter++ large is close enough to be a serious paper baseline, especially because monodepth priors may help with low-texture tissue. RAFT-Stereo and CREStereo are useful reviewer-proof baselines, but are weaker on current SERV-CT depth metrics.
