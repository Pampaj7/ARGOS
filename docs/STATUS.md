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
- Started DEFOM-Stereo checkpoint download in detached `screen` session `argos_defom_download`.
- Added combined SERV-CT scoreboard:
  - `stereo/argos_baselines/docs/servct_scoreboard.md`;
  - `stereo/argos_baselines/images/servct_depth_mae_scoreboard.png`;
  - mirrored into ARGOS as `docs/SERVCT_BASELINE_SCOREBOARD.md`.
- Restarted long downloads in detached `screen` sessions after detecting stale network sockets:
  - `argos_scared_download`;
  - `argos_training_extras`;
  - `argos_monsterpp_large`.
- Initialized local ARGOS git repo with README, metrics, result images, and scripts. GitHub push is waiting for `gh auth login`.

## Current Tasks

- Finish SCARED download and write SCARED converter to dense disparity/depth training format.
- Finish DEFOM-Stereo download and evaluate on SERV-CT.
- Add S2M2-L/XL when the extra training download queue reaches those weights.
- Add IGEV++/Selective-Stereo when checkpoints are available.
- Prepare a larger S2M2 fine-tuning run using SERV-CT + SCARED.
- Publish lightweight ARGOS repo to GitHub once authentication is available.

## GitHub

Local repo: `/home/pampaj/Desktop/ARGOS`

Push command after authentication:

```bash
cd /home/pampaj/Desktop/ARGOS
gh auth login
gh repo create ARGOS --public --source=. --remote=origin --push
```
