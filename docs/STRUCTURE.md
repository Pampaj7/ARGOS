# Stereo Workspace Structure

This file is the local map for `/home/pampaj/Desktop/stereo`.

## Main Buckets

| Bucket | What Belongs Here | Notes |
|---|---|---|
| `results/` | ARGOS-owned reports, JSON/CSV metrics, plots, montages, smoke-test logs | Keep this relatively small and portable. |
| `models/` | Symlink hub for easier navigation by model type | Does not own repo contents; links back to top-level repos. |
| `download_jobs/` | Download scripts, queues, and logs | Useful for resuming or auditing slow dataset downloads. |
| `argos_baselines/` | Local baseline helper scripts, scoreboards, summary docs | Older ARGOS helper area before the clean `/ARGOS` repo absorbed most scripts. |
| `argos_data/` | Small converted ARGOS-ready data samples/subsets | Good for reproducible smoke tests. |
| `datasets/` | Raw or externally downloaded datasets | Can be huge and incomplete. |
| top-level model dirs | Cloned third-party model repos | Kept in place so each baseline is visible and scripts have stable paths. |

## Top-Level Model Repos

For human navigation, use the symlink hub:

```text
models/
  frame_stereo/
  video_stereo/
```

The real repo directories remain at top level:

| Directory | Type | Role |
|---|---|---|
| `Fast-FoundationStereo/` | Frame stereo | Fast-FoundationStereo baseline, SCARED/ServCT experiments, ONNX and output artifacts. |
| `s2m2/` | Frame stereo | S2M2 baseline and fine-tuning experiments. |
| `MonSter-plusplus/` | Frame stereo | MonSter++ and RT-MonSter++ baseline tests. |
| `DEFOM-Stereo/` | Frame stereo | DEFOM-Stereo baseline tests. |
| `RAFT-Stereo/` | Frame stereo | RAFT-Stereo baseline tests. |
| `stereoanywhere/` | Frame stereo | StereoAnywhere baseline tests. |
| `stereo_matching_crestereo/` | Frame stereo | CREStereo package/baseline tests. |
| `FoundationStereo/` | Frame stereo | Original FoundationStereo repo. |
| `IGEV-plusplus/` | Frame stereo | IGEV++ repo. |
| `Selective-Stereo/` | Frame stereo | Selective RAFT/IGEV repo. |
| `stereoanyvideo/` | Video stereo | Stereo Any Video temporal baseline and ARGOS smoke outputs. |
| `TemporalStereo/` | Video stereo | TemporalStereo repo for sequence stereo scouting. |
| `Temporally-Consistent-Stereo-Matching/` | Video stereo | TC-Stereo repo for geometry-aware temporal stereo scouting. |
| `PPMStereo/` | Video stereo | PPMStereo exploratory temporal baseline. |
| `TemporallyConsistentDepth/` | Temporal depth | Optional temporal depth consistency baseline, not pure stereo. |

## Output Conventions

Use this pattern for new work:

```text
results/
  <experiment_or_scout_name>/
    report.md
    report.json
    metrics/
    images/
    logs/
    smoke_inputs/
```

When a third-party model writes inside its own repo, copy or summarize the final portable artifacts into `results/`.

## Dependency Conventions

Do not install global dependencies from third-party repos. Put env notes under `results/<experiment>/env_notes/<model>.yml` or add the env command in the relevant experiment report.

## What To Avoid

- Do not mix raw datasets with reports.
- Do not save new checkpoints under `results/`.
- Do not hide active baseline repos in nested folders.
- Do not delete or move existing large model repos casually; many scripts still expect their current location.
