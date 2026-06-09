# ARGOS Status Log

## 2026-06-08

- Reorganized all local data under `ARGOS/dataset/`:
  - raw SCARED archives: `dataset/raw/surgical_stereo/scared/`;
  - raw SERV-CT: `dataset/raw/surgical_stereo/servct/`;
  - EndoSLAM/support data: `dataset/raw/external_datasets/`;
  - processed workspace data: `dataset/workspace_argos_data/`.
- Added curated ready-to-use subsets:
  - `dataset/scared_consecutive32/`;
  - `dataset/scared_rect5/`;
  - `dataset/servct_argos/`.
- Ran S2M2-S honest SERV-CT fine-tuning on GPU:
  - train: Experiment_1 / Reference_CT;
  - test: Experiment_2 / Reference_CT;
  - mode: refiners-only, 250 steps, LR `2e-5`.
- Result on Experiment_2:
  - pretrained depth MAE: `1.7638 mm`;
  - fine-tuned depth MAE: `1.4580 mm`;
  - pretrained disparity MAE: `1.4615 px`;
  - fine-tuned disparity MAE: `1.4684 px`.
- Added before/after error montage and report:
  - `results/servct_s2m2_honest_finetune_gpu/report.md`;
  - `results/servct_s2m2_honest_finetune_gpu/s2m2_servct_before_after_error_montage.png`.
- Tested SERV-CT fine-tuned S2M2-S transfer to rectified SCARED dataset_8 keyframes:
  - pretrained S2M2-S depth MAE: `2.8372 mm`;
  - SERV-CT fine-tuned depth MAE: `2.8320 mm`;
  - pretrained disparity MAE: `4.2431 px`;
  - SERV-CT fine-tuned disparity MAE: `4.2832 px`;
  - conclusion: nearly neutral transfer, so mixed SERV-CT + SCARED tuning is likely needed.

## 2026-06-03

- Created `stereo/` workspace with multiple upstream stereo models.
- Evaluated Fast-FoundationStereo ONNX, S2M2-S/M, Stereo Anywhere, and SGBM on SERV-CT.
- Fine-tuned S2M2-S on SERV-CT with two regimes:
  - honest holdout: train Experiment_1 CT, eval Experiment_2 CT;
  - all surgical: train SERV-CT CT/RGB references, eval CT.
- Started full SCARED download in detached `screen` session `argos_scared_download`.
- Started extra download queue in detached `screen` session `argos_training_extras`.
- Cloned MonSter++ and prepared common SERV-CT evaluation script.
- Evaluated RT-MonSter++ `Zero_shot.pth` on SERV-CT:
  - disparity MAE: 1.60 px;
  - disparity RMSE: 2.78 px;
  - depth MAE: 2.05 mm;
  - depth RMSE: 3.42 mm;
  - frames: 16.
- Downloaded and evaluated MonSter++ `Mix_all_large.pth` on SERV-CT.
- Cloned and evaluated CREStereo on SERV-CT.
- Cloned RAFT-Stereo, downloaded Dropbox pretrained models, and evaluated RVC/Middlebury checkpoints on SERV-CT.
- Cloned IGEV++ and Selective-Stereo; both are waiting on Google Drive/manual checkpoint access.
- Downloaded DEFOM-Stereo checkpoints and evaluated VITS SceneFlow, VITS RVC, and VIT-L Middlebury on SERV-CT.
  - best DEFOM run so far: VIT-L Middlebury, disparity MAE 1.73 px, depth MAE 1.99 mm;
  - VIT-L SceneFlow was unstable on SERV-CT depth and is not used as a primary score.
- Added combined SERV-CT scoreboard:
  - `stereo/argos_baselines/docs/servct_scoreboard.md`;
  - `stereo/argos_baselines/images/servct_depth_mae_scoreboard.png`;
  - mirrored into ARGOS as `docs/SERVCT_BASELINE_SCOREBOARD.md`.
- Added paper/repo scaffolding:
  - `docs/EXPERIMENT_PROTOCOL.md`;
  - `docs/DATASETS.md`;
  - `docs/MODEL_ZOO.md`;
  - `docs/ROADMAP.md`;
  - `configs/servct_baselines.yaml`;
  - `configs/surgical_splits.yaml`.
- Added SERV-CT converter and verified it creates 16 unified ARGOS samples under `/home/pampaj/Desktop/stereo/argos_data/servct/`.
- Added SCARED converter placeholder that reports available archives until the full dataset layout is available.
- Added `scripts/run_all_servct_baselines.py` for lightweight regeneration of current scoreboard outputs.
- Integrated the ARGOS-Wound internal proposal direction into the main README:
  - v0 geometry-anchored stereo/RGB benchmark;
  - v1 active depth / ToF / LiDAR extension;
  - failure-aware ARGOS-Fuse direction;
  - open-wound ex-vivo acquisition and MICCAI 2027 positioning.
- Restarted long downloads in detached `screen` sessions after detecting stale network sockets:
  - `argos_scared_download`;
  - `argos_training_extras`;
  - `argos_monsterpp_large`.
- Initialized local ARGOS git repo with README, metrics, result images, and scripts. GitHub push is waiting for `gh auth login`.

## Current Tasks

- Finish SCARED download and write SCARED converter to dense disparity/depth training format.
- Extend the SCARED converter once the full archive layout is available.
- Add S2M2-L/XL when the extra training download queue reaches those weights.
- Add IGEV++/Selective-Stereo when checkpoints are available.
- Prepare a larger S2M2 fine-tuning run using SERV-CT + SCARED.
- Publish lightweight ARGOS repo to GitHub once authentication is available.

## 2026-06-09

- Benchmarked S2M2-S, S2M2-L, and S2M2-XL on rectified SCARED dataset_8 clean keyframes.
- Evaluated full resolution plus resized widths `1024`, `736`, and `512`, with disparity predictions rescaled back to original image coordinates.
- Result folder: `results/s2m2_size_tradeoff/`.
- Best current result:
  - XL full: `2.6963 mm` depth MAE, `4.1303 px` disparity MAE;
  - L full: `2.7100 mm` depth MAE, `4.1445 px` disparity MAE;
  - S full: `2.7452 mm` depth MAE, `4.1667 px` disparity MAE.
- Interpretation: XL wins, but the margin over L is tiny relative to cost; use XL as teacher candidate, L as balanced default baseline, and S/512 as the fastest real-time candidate.

- Extended the S2M2 benchmark into a resolution tradeoff study:
  - result folder: `results/s2m2_resolution_tradeoff/`;
  - best S/L under 500 ms: L full, `2.7100 mm` depth MAE, `485.10 ms`;
  - best S/L under 300 ms: L@736, `2.7425 mm` depth MAE, `185.86 ms`;
  - lowest VRAM S/L option: S@512, `2.8660 mm` depth MAE, `374.9 MB` VRAM;
  - current deployment recommendation: L@736, with XL full reserved as teacher/reference.

- Added video-stereo repository scouting under `external/video_stereo_repos/` and `results/video_stereo_repos/`.
- Cloned/linked:
  - TemporalStereo;
  - TC-Stereo / Temporally Consistent Stereo Matching;
  - DynamicStereo;
  - BiDAStereo;
  - StereoAnyVideo.
- Prepared a common 5-frame rectified SCARED dataset_8 smoke-test package with left/right images, GT disparity, GT depth, and valid masks.
- Smoke-test status:
  - StereoAnyVideo runs on ARGOS/SCARED custom stereo folders with local checkpoint and CUDA;
  - TemporalStereo needs isolated Python 3.8/PyTorch 1.10/CUDA 11.3 plus Apex/Detectron2/Cupy and checkpoint;
  - TC-Stereo needs separate env plus Dropbox checkpoint;
  - DynamicStereo needs Hydra/PyTorch3D/checkpoint and has CC-BY-NC-4.0 license;
  - BiDAStereo needs Hydra/PyTorch3D/checkpoint and likely high VRAM.
- StereoAnyVideo smoke metrics on the 5-frame SCARED package:
  - disparity MAE: `4.1624 px`;
  - depth MAE: `2.7090 mm`;
  - bad-2mm: `18.81%`;
  - comparable to S2M2-L full on this small smoke subset, but true temporal conclusions need consecutive video frames.

- Integrated StereoAnyVideo as the first video-stereo upper-bound baseline:
  - result folder: `results/stereoanyvideo_temporal_eval/`;
  - compared against S2M2-L@full, S2M2-L@736, and S2M2-S@512;
  - used `gt5` for GT accuracy and `consecutive32` for true temporal/flicker metrics.
- Key StereoAnyVideo temporal result on `consecutive32`:
  - mean consecutive disparity diff: `1.0214`;
  - S2M2-L@736: `1.6737`;
  - S2M2-S@512: `1.2217`;
  - S2M2-L@full: `8.6276`.
- Key accuracy result on `gt5`:
  - StereoAnyVideo@384x640 depth MAE: `2.7090 mm`;
  - S2M2-L@full depth MAE: `2.7100 mm`;
  - S2M2-L@736 depth MAE: `2.7425 mm`.
- Interpretation: StereoAnyVideo is a strong temporal teacher/reference and video quality upper bound. It is not yet the deployment default because the 32-frame run uses about `10.1 GB` peak VRAM at 384x640.

## GitHub

Local repo: `/home/lpampaloni/ARGOS`

Push command after authentication:

```bash
cd /home/lpampaloni/ARGOS
gh auth login
gh repo create ARGOS --public --source=. --remote=origin --push
```
