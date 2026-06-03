# ARGOS-Wound Surgical Perception

ARGOS is a surgical perception research workspace for open-wound 3D reconstruction, semantic understanding, and failure-aware multimodal fusion. The current repository contains the first technical pillar of that effort: a controlled stereo-depth benchmark and adaptation pipeline for close-range surgical tissue scenes.

The broader target is **ARGOS-Wound**: a staged, release-ready benchmark for open-surgery wound perception using ex-vivo porcine tissue, calibrated multi-view sensing, static anchor-state reference geometry, semantic annotations, wound-edge labels, tool/hand/occlusion labels, and uncertainty-aware evaluation for future robotic surgical guidance.

## Current Direction

The current v0 direction is to de-risk the geometric perception layer before moving to more complex autonomy modules. We evaluate modern stereo models on surgical stereo with ground truth, convert datasets into a unified ARGOS format, then fine-tune the strongest model family on surgical data.

Primary benchmark so far:

- **SERV-CT**: rectified stereo endoscopy with CT/RGB-scan reference disparity and depth.
- **SCARED**: currently downloading full dataset for larger supervised surgical training.
- **EndoSLAM**: queued as support data for domain expansion, pose/3D validation, and possible pseudo-labeling.

Planned ARGOS-Wound benchmark direction:

- **v0**: geometry-anchored stereo/RGB benchmark with static reference scans and core semantic labels.
- **v1**: active depth and ToF/LiDAR extension, evaluated against anchor-state reference geometry.
- **v2**: longer-term manipulation/contact-aware extension with tool trajectories, tool-tip localization, and suturing primitives.

## Project Thesis

ARGOS is being shaped as both a codebase and a paper around two nested questions:

1. Can foundation-scale stereo models be adapted into reliable millimetric depth estimators for close-range surgical wound scenes?
2. Can semantic and uncertainty-aware multimodal fusion detect when and where open-wound reconstruction should not be trusted?

The working hypothesis is that generic stereo/foundation models already contain useful geometry priors, but open surgical perception needs domain adaptation, careful split discipline, semantic masking, and explicit uncertainty/failure detection. ARGOS therefore combines:

- a surgical stereo benchmark with ground-truth depth/disparity;
- a broad zero-shot baseline suite across classical, recurrent, foundation, scalable, and monodepth-prior stereo models;
- a unified dataset format for SERV-CT, SCARED, and future surgical stereo sources;
- honest surgical fine-tuning protocols that separate real generalization from all-data adaptation;
- qualitative figures and failure analysis for wounds, tissue surfaces, instruments, textureless areas, and specular regions;
- a future ARGOS-Fuse direction: fused visible wound-surface reconstruction, geometry uncertainty maps, and unsafe-region masks.

The current lead model family is S2M2, with MonSter++ large and Fast-FoundationStereo as strong zero-shot comparators.

## ARGOS-Wound Vision

The longer-term benchmark is designed around controlled ex-vivo open-surgery scenes rather than endoscopic-only perception. The intended acquisition setup includes:

- porcine tissue wound surrogates with visible skin/fat/fascia-like/muscle layers;
- synchronized stereo/RGB or RGB-D streams;
- a global context camera for hands, tools, occlusion, and workflow;
- rigid fiducials for calibration into a shared metric coordinate frame;
- static anchor-state reference geometry from structured-light or high-accuracy scanning;
- semantic masks for wound, wound edge, tissue, hand/glove, tool, gauze, blood/fluid, background, and occlusion.

The first benchmark version will not overclaim dense dynamic ground truth for deforming tissue. Instead, it will combine continuous synchronized sensing streams with high-quality static anchor geometry at selected deformation states. Dynamic clips will be evaluated through temporal consistency, visible-surface stability, occlusion recovery, and sensor failure-mode analysis.

The methodological direction is **failure-aware open-wound perception**. A surgical perception system should not only reconstruct the visible wound surface; it should also estimate where the reconstruction is unreliable, incomplete, occluded, or unsafe for downstream robotic guidance.

## Models Tested

| Model | Status | Notes |
|---|---|---|
| Fast-FoundationStereo ONNX | tested | Strong zero-shot baseline on SERV-CT. |
| S2M2-S / S2M2-M | tested | Best current candidate; S2M2-S wins most surgical metrics. |
| Stereo Anywhere | tested | Works with Depth Anything V2 prior; weaker than S2M2 on SERV-CT. |
| OpenCV SGBM | tested | Classical baseline, much worse than learned models. |
| FoundationStereo full | blocked | Code cloned, Google Drive weights quota-blocked. |
| DEFOM-Stereo | tested | VIT-L Middlebury, VITS RVC, and VITS SceneFlow evaluated on SERV-CT. |
| RT-MonSter++ | tested | Zero-shot real-time checkpoint evaluated on SERV-CT. |
| MonSter++ large | tested | MixAll large checkpoint evaluated on SERV-CT. |
| RAFT-Stereo | tested | RVC and Middlebury checkpoints evaluated on SERV-CT. |
| CREStereo | tested | Bundled checkpoint evaluated on SERV-CT. |
| IGEV++ | cloned | Awaiting Google Drive/manual weights. |
| Selective-Stereo | cloned | Awaiting Google Drive/manual weights. |

## Best Numbers So Far

S2M2-S fine-tuning on SERV-CT:

| Run | Disp MAE | Depth MAE | Scope |
|---|---:|---:|---|
| Baseline S2M2-S | 1.46 px | 1.76 mm | pretrained, eval Experiment_2 CT |
| Honest holdout tune | 1.38 px | 1.37 mm | train Experiment_1 CT, eval Experiment_2 CT |
| All surgical tune | 0.81 px | 1.02 mm | train CT+RGB refs, eval CT |
| DEFOM-Stereo VIT-L Middlebury | 1.73 px | 1.99 mm | pretrained zero-shot, full SERV-CT CT |
| RT-MonSter++ zero-shot | 1.60 px | 2.05 mm | pretrained zero-shot, full SERV-CT CT |

The all-surgical run is a domain-adapted checkpoint, not an independent holdout metric.

## Planned Paper Structure

1. **Problem**: open-wound perception is close-range, deformable, wet, specular, occluded by hands/tools, and safety-critical.
2. **Phase 1 de-risking**: determine whether open-wound perception can be calibrated, annotated, reconstructed, and evaluated reliably under controlled ex-vivo conditions.
3. **v0 benchmark**: establish geometry-anchored stereo/RGB evaluation with static reference scans, fiducials, semantic labels, and visible-surface metrics.
4. **Stereo foundation baselines**: SGBM, RAFT-Stereo, CREStereo, Fast/FoundationStereo, Stereo Anywhere, MonSter++, DEFOM-Stereo, and S2M2 variants.
5. **Adaptation**: fine-tune the strongest scalable stereo family on surgical data with honest train/test and cross-dataset splits.
6. **ARGOS-Fuse**: develop semantic and uncertainty-aware fusion for visible wound-surface reconstruction and unsafe-region detection.
7. **Analysis**: report metric depth error, boundary/detail behavior, near-field performance, temporal consistency, failure cases, and qualitative surgical montages.
8. **Outcome**: establish a reproducible perception benchmark and method-development path toward a MICCAI 2027 ARGOS-Wound submission.

## Results

Key images are kept in `results/images/`:

- `argos_surgical_stereo_model_comparison.png`: zero-shot model comparison.
- `argos_s2m2_finetune_results.png`: S2M2 surgical fine-tuning results.
- `argos_s2m2_comparison.png`: S2M2-S/M vs previous baseline.
- `argos_stereo_surgical_results.png`: early Fast-FoundationStereo/SERV-CT/SCARED summary.
- `rtmonsterplusplus_zeroshot_servct_montage.png`: RT-MonSter++ zero-shot SERV-CT montage.
- `servct_depth_mae_scoreboard.png`: current SERV-CT depth-MAE baseline ranking.

Full baseline table: `docs/SERVCT_BASELINE_SCOREBOARD.md`.

## Protocol And Project Docs

- `docs/ARGOS.pdf`: internal ARGOS-Wound proposal and broader benchmark/method direction.
- `docs/EXPERIMENT_PROTOCOL.md`: zero-shot, honest fine-tune, all-surgical adaptation, and cross-dataset rules.
- `docs/DATASETS.md`: dataset status and target unified ARGOS sample format.
- `docs/MODEL_ZOO.md`: tested and pending model baselines.
- `docs/ROADMAP.md`: project phases toward paper-ready experiments.
- `configs/servct_baselines.yaml`: current SERV-CT benchmark metadata.
- `configs/surgical_splits.yaml`: current surgical train/test split definitions.

SERV-CT has a working converter to the unified local format under `/home/pampaj/Desktop/stereo/argos_data/servct/`. Converted data is intentionally excluded from git.

## Immediate Next Steps

- Finish SCARED download and implement the real SCARED converter.
- Run S2M2-L/XL zero-shot once the queued weights arrive.
- Fine-tune S2M2 on SERV-CT + SCARED using the protocol in `docs/EXPERIMENT_PROTOCOL.md`.
- Add cross-dataset evaluation: SERV-CT to SCARED and SCARED to SERV-CT.
- Add surgical robustness metrics: near-field bins, boundary/detail masks, and specular/textureless failure slices.
- Turn current scoreboards and montages into stable paper figures.
- Translate the ARGOS-Wound proposal into concrete v0 acquisition requirements: sensor list, fiducial layout, anchor-state scan protocol, semantic label set, and release metadata.
- Draft the ARGOS-Fuse evaluation target: fused surface, uncertainty map, and unsafe-region mask.

## Active Downloads

Two detachable `screen` sessions are used on the workstation:

```bash
screen -ls
tail -f /home/pampaj/Desktop/stereo/download_jobs/scared_full_download.log
tail -f /home/pampaj/Desktop/stereo/download_jobs/training_extras_download.log
```

The extra queue waits for SCARED to finish, then downloads S2M2-L/XL and EndoSLAM.

MonSter++ and DEFOM-Stereo are also set up locally under `/home/pampaj/Desktop/stereo/`; upstream repos and model weights are intentionally excluded from this ARGOS repository.

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
