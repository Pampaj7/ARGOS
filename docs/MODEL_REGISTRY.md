# ARGOS Stereo Model Registry

This registry tracks which model repos exist in the local stereo workspace and how we intend to use them.

## Frame-Based / Single-Pair Stereo Baselines

| Model | Local Path | Status | Notes |
|---|---|---|---|
| Fast-FoundationStereo | `Fast-FoundationStereo/` | Active baseline | Strong frame-based surgical baseline; contains large local data/output artifacts. |
| S2M2 | `s2m2/` | Active baseline and fine-tuning target | Main fine-tuning direction so far. |
| MonSter++ / RT-MonSter++ | `MonSter-plusplus/` | Active baseline | Useful strong modern baseline with monodepth priors. |
| DEFOM-Stereo | `DEFOM-Stereo/` | Tested baseline | Large local outputs/checkpoints. |
| RAFT-Stereo | `RAFT-Stereo/` | Tested classic baseline | Useful sanity baseline. |
| StereoAnywhere | `stereoanywhere/` | Tested baseline | Surgical zero-shot experiments present. |
| CREStereo | `stereo_matching_crestereo/` | Tested baseline | Package-style repo. |
| FoundationStereo | `FoundationStereo/` | Reference baseline | Original repo kept for comparison/reference. |
| IGEV++ | `IGEV-plusplus/` | Candidate/reference | Strong stereo family. |
| Selective-Stereo | `Selective-Stereo/` | Candidate/reference | Selective RAFT/IGEV variants. |

## Video / Temporally Consistent Stereo

Detailed scouting report:

`results/video_stereo_repo_scouting/report.md`

| Model | Local Path | Status | Recommended Role |
|---|---|---|---|
| Stereo Any Video | `stereoanyvideo/` | Partial clone, inference files inspected | First video-native integration target. |
| TC-Stereo | `Temporally-Consistent-Stereo-Matching/` | Partially materialized, metadata inspected | Second integration target if pose/intrinsics are available. |
| TemporalStereo | `TemporalStereo/` | Shallow clone inspected | Fallback temporal baseline if TC integration blocks. |
| PPMStereo | `PPMStereo/` | Commit recorded, raw metadata inspected | Recent exploratory baseline; likely GPU-heavy. |
| TemporallyConsistentDepth | `TemporallyConsistentDepth/` | Commit recorded, raw metadata inspected | Optional temporal depth fusion baseline, not pure stereo. |

## Recommendation Right Now

1. Integrate Stereo Any Video into a SCARED temporal smoke pipeline.
2. Integrate TC-Stereo only after deciding how to provide pose/intrinsics/baseline for surgical sequences.
3. Keep PPMStereo for a later high-memory GPU pass.

## Active Smoke Tests

| Model | Report | Status |
|---|---|---|
| Stereo Any Video | `results/stereoanyvideo_scared_smoke/` | Completed 5-frame SCARED smoke test with local checkpoints. |
| Stereo Any Video vs S2M2-S vs Fast-FoundationStereo | `results/temporal_context_comparison_scared5/report.md` | Completed smoke comparison on non-consecutive keyframes; not used for temporal claims. |
| Stereo Any Video vs S2M2-S vs Fast-FoundationStereo | `results/temporal_context_comparison_scared_consecutive32/report.md` | Completed corrected temporal comparison on 32 consecutive SCARED video frames. |
