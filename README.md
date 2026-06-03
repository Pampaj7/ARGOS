# ARGOS Surgical Stereo Depth

ARGOS is an experimental benchmark and adaptation workspace for close-range surgical stereo depth estimation. The current goal is to find and adapt stereo models that can recover accurate millimetric depth on endoscopic scenes such as surgical wounds, tissue surfaces, and instruments.

## Current Direction

We evaluate modern stereo models on surgical stereo with ground truth, then fine-tune the best candidate on surgical data.

Primary benchmark so far:

- **SERV-CT**: rectified stereo endoscopy with CT/RGB-scan reference disparity and depth.
- **SCARED**: currently downloading full dataset for larger supervised surgical training.
- **EndoSLAM**: queued as support data for domain expansion, pose/3D validation, and possible pseudo-labeling.

## Models Tested

| Model | Status | Notes |
|---|---|---|
| Fast-FoundationStereo ONNX | tested | Strong zero-shot baseline on SERV-CT. |
| S2M2-S / S2M2-M | tested | Best current candidate; S2M2-S wins most surgical metrics. |
| Stereo Anywhere | tested | Works with Depth Anything V2 prior; weaker than S2M2 on SERV-CT. |
| OpenCV SGBM | tested | Classical baseline, much worse than learned models. |
| FoundationStereo full | blocked | Code cloned, Google Drive weights quota-blocked. |
| DEFOM-Stereo | cloned | Awaiting weight/dependency follow-up. |
| MonSter++ | in progress | Repo cloned; Mix_all_large checkpoint downloading; common SERV-CT benchmark prepared. |

## Best Numbers So Far

S2M2-S fine-tuning on SERV-CT:

| Run | Disp MAE | Depth MAE | Scope |
|---|---:|---:|---|
| Baseline S2M2-S | 1.46 px | 1.76 mm | pretrained, eval Experiment_2 CT |
| Honest holdout tune | 1.38 px | 1.37 mm | train Experiment_1 CT, eval Experiment_2 CT |
| All surgical tune | 0.81 px | 1.02 mm | train CT+RGB refs, eval CT |

The all-surgical run is a domain-adapted checkpoint, not an independent holdout metric.

## Results

Key images are kept in `results/images/`:

- `argos_surgical_stereo_model_comparison.png`: zero-shot model comparison.
- `argos_s2m2_finetune_results.png`: S2M2 surgical fine-tuning results.
- `argos_s2m2_comparison.png`: S2M2-S/M vs previous baseline.
- `argos_stereo_surgical_results.png`: early Fast-FoundationStereo/SERV-CT/SCARED summary.

## Active Downloads

Two detachable `screen` sessions are used on the workstation:

```bash
screen -ls
tail -f /home/pampaj/Desktop/stereo/download_jobs/scared_full_download.log
tail -f /home/pampaj/Desktop/stereo/download_jobs/training_extras_download.log
```

The extra queue waits for SCARED to finish, then downloads S2M2-L/XL and EndoSLAM.

MonSter++ is also being set up locally under `/home/pampaj/Desktop/stereo/MonSter-plusplus`; its upstream repo and model weights are intentionally excluded from this ARGOS repository.

## Repository Policy

This repo intentionally does **not** include downloaded upstream repositories, model weights, or datasets. It tracks only:

- project notes and README,
- evaluation/fine-tuning scripts,
- compact result images,
- small metric summaries/configs.

Large data lives locally under `/home/pampaj/Desktop/stereo/`.

## GitHub Publishing

The local ARGOS git repo is initialized. GitHub CLI is installed but not authenticated on this machine yet. To publish:

```bash
cd /home/pampaj/Desktop/ARGOS
gh auth login
gh repo create ARGOS --public --source=. --remote=origin --push
```
