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
- Large MonSter++ checkpoint follow-up is still pending.
- Restarted long downloads in detached `screen` sessions after detecting stale network sockets:
  - `argos_scared_download`;
  - `argos_training_extras`;
  - `argos_monsterpp_large`.
- Initialized local ARGOS git repo with README, metrics, result images, and scripts. GitHub push is waiting for `gh auth login`.

## Current Tasks

- Finish SCARED download and write SCARED converter to dense disparity/depth training format.
- Try large MonSter++ checkpoint on SERV-CT when the large weights are available.
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
