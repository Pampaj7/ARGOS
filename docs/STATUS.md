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
- Started downloading MonSter++ `Mix_all_large.pth` checkpoint from Hugging Face.
- Initialized local ARGOS git repo with README, metrics, result images, and scripts. GitHub push is waiting for `gh auth login`.

## Current Tasks

- Finish SCARED download and write SCARED converter to dense disparity/depth training format.
- Benchmark MonSter++ on SERV-CT.
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
