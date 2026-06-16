# Video Stereo Repository Scouting

Repos are isolated under `external/video_stereo_repos/`. No global dependency installation was performed.

Common smoke-test sequence: `results/video_stereo_repos/test_sequence/` with 5 rectified SCARED dataset_8 keyframes, left/right images, GT disparity, GT depth, valid masks, and metadata.

Important caveat: these 5 clean keyframes are not guaranteed to be a temporally consecutive video clip. The smoke test checks integration and custom input handling first; true temporal behavior still needs a consecutive SCARED warped/clean sequence.

## Repo Table

| model | commit | license | checkpoint | custom sequence | pose | flow | intrinsics | difficulty | smoke | next action |
|---|---|---|---|---|---|---|---|---|---|---|
| TemporalStereo | `0b5a94eb` | Apache-2.0 | manual Google Drive | possible with wrapper | yes for video_inference path; maybe no for dataset demo | no documented requirement | yes/likely for video mode | high | failed: matplotlib style/env mismatch | Create isolated temporalstereo conda env; download checkpoint; patch matplotlib style or install compatible seaborn; then write ARGOS folder-sequence wrapper. |
| TC-Stereo | `unknown` | MIT | manual Dropbox | possible with custom dataset wrapper | yes for temporal consistency training/eval datasets | no documented requirement | likely for depth/pose-aware evaluation; disparity inference itself may not | medium-high | failed: missing wandb in shared env | Download Dropbox checkpoint and build minimal ARGOS eval_stereo wrapper; likely best first non-SAV integration if checkpoint retrieval succeeds. |
| DynamicStereo | `c5077aa1` | CC-BY-NC-4.0 | automatic/manual from README | possible through real-data config or wrapper | yes/likely; dataset provides intrinsics/extrinsics and depth | no for inference, but dataset includes flow/trajectories | yes for depth and Dynamic Replica format | medium-high | failed: missing hydra in shared env | Create isolated dynamicstereo env and download real-data checkpoint; evaluate real config on ARGOS sequence if intrinsics format can be matched. |
| BiDAStereo | `22c405d9` | MIT | manual GitHub release | possible through real-data config/wrapper | yes/likely for Dynamic Replica; real-data may be less strict | alignment method may use RAFT/flow components; not a simple frame-only stereo CLI | likely for depth/eval; disparity model itself can operate on stereo pairs | high | failed: missing hydra in shared env | Defer until TC-Stereo/DynamicStereo are sorted; useful as quality-oriented recent baseline, not deployment first. |
| StereoAnyVideo | `3a8beddc` | Apache-2.0 | yes, local | yes | no | no explicit external flow | no for disparity inference; needed only to convert to metric depth | low-medium | success on ARGOS/SCARED test_sequence using CUDA | Integrate first as video quality upper bound and add timed wrapper at 384x640/512x736/720x1280. |

## Smoke Test Results

| model | disp MAE | depth MAE | depth median | bad 2px | bad 2mm | failure <=0.5 | temporal pred diff | temporal error variation | runtime ms | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| StereoAnyVideo@384x640 | 4.1624 | 2.7090 | 0.6353 | 15.86 | 18.81 | 0.0000 | 16.9937 | 6.6481 |  |  |
| S2M2-L@736 | 4.1709 | 2.7425 | 0.6919 | 15.83 | 19.67 | 0.0000 | 17.0442 | 6.6196 | 324.13 | 1672.5 |
| S2M2-L@full | 4.1445 | 2.7100 | 0.6542 | 15.47 | 18.99 | 0.0000 | 17.0243 | 6.6162 | 488.60 | 2897.7 |
| S2M2-S@512 | 4.2772 | 2.8660 | 0.8206 | 18.01 | 22.23 | 0.0000 | 17.0804 | 6.6521 | 74.98 | 371.3 |

StereoAnyVideo smoke command and logs are in `results/video_stereo_repos/StereoAnyVideo/`. Other repos have failed smoke logs in their respective folders; failures are dependency/checkpoint integration findings, not model-quality conclusions.

## Answers

1. Runnable now: StereoAnyVideo is runnable on the ARGOS/SCARED custom sequence with the local checkpoint. TemporalStereo, TC-Stereo, DynamicStereo, and BiDAStereo are cloned but not runnable in the shared env without isolated dependency setup and/or checkpoints.
2. Minimal custom rectified sequence support: StereoAnyVideo accepts `left/` and `right/` image folders directly. DynamicStereo and BiDAStereo have real-data configs and should be adaptable. TC-Stereo and TemporalStereo need custom dataset/wrapper work and likely pose/intrinsics handling.
3. Usable pretrained checkpoints: StereoAnyVideo has a local checkpoint. DynamicStereo has scripted/manual checkpoint download. TC-Stereo and TemporalStereo provide Dropbox/Google Drive checkpoint links. BiDAStereo provides GitHub release checkpoints.
4. First ARGOS integration: StereoAnyVideo first, because it already runs on custom stereo folders and gives a quality upper-bound style temporal baseline. Next should be TC-Stereo if its Dropbox checkpoint is easy to retrieve; otherwise DynamicStereo real-data config.
5. Model roles:
   - video quality upper bound: StereoAnyVideo;
   - efficient temporal baseline: TC-Stereo candidate once checkpoint/env are ready; TemporalStereo is older but efficient on paper;
   - deployment candidate: none proven yet; compare TC-Stereo/TemporalStereo against S2M2-L@736 after real smoke runs;
   - temporal distillation teacher: StereoAnyVideo now, possibly BiDAStereo/DynamicStereo after successful checkpoint setup.

## Main Blockers

- TemporalStereo: Python 3.8/PyTorch 1.10/CUDA 11.3 plus Apex, Detectron2, Cupy; smoke currently fails on matplotlib style before deeper imports.
- TC-Stereo: missing `wandb` in shared env and no checkpoint present; likely manageable in a separate env.
- DynamicStereo: missing Hydra/PyTorch3D stack and checkpoint; CC-BY-NC-4.0 license should be noted for downstream use.
- BiDAStereo: missing Hydra/PyTorch3D/checkpoint and documented high VRAM expectation.
- StereoAnyVideo: runs, but needs timed/VRAM-instrumented wrapper and a true consecutive SCARED video sequence for temporal claims.
