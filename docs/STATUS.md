# ARGOS Status Log

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

## GitHub

Local repo: `/home/lpampaloni/ARGOS`

Push command after authentication:

```bash
cd /home/lpampaloni/ARGOS
gh auth login
gh repo create ARGOS --public --source=. --remote=origin --push
```
