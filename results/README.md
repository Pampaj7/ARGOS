# ARGOS Results Directory

This folder contains generated experiment outputs. Large payloads are local evidence and are not meant to be committed.

The rule is: keep the root clean and put every result package under one category.

```text
results/
  servct evaluation/
  scared evaluation/
  01_frame_stereo/
  02_video_stereo/
  03_temporal_refinement/
  04_dataset_derivatives/
  90_legacy/
  README.md
```

## Category Map

| Folder | Purpose |
|---|---|
| `servct evaluation/` | single SERV-CT baseline table for SOTA model comparison |
| `scared evaluation/` | single SCARED long-subset inference table for available model streams |
| `01_frame_stereo/` | frame-by-frame stereo benchmarks and fine-tuning checks |
| `02_video_stereo/` | video-stereo baselines and StereoAnyVideo comparisons |
| `03_temporal_refinement/` | ARGOS temporal-refinement design, caches, training, and evaluation |
| `04_dataset_derivatives/` | generated dataset-derived artifacts: audits, long-sequence extracts, frozen predictions, and logs |
| `90_legacy/` | parking area for obsolete outputs if needed |

## Current Canonical Evidence

| Question | Start Here |
|---|---|
| Presentation assets | `../presentation/argos_progress/` |
| SERV-CT baseline table | `servct evaluation/servct_evaluation.md` |
| SCARED long-subset inference table | `scared evaluation/scared_evaluation.md` |
| SERV-CT frame-stereo benchmark | `01_frame_stereo/SERVCT/servct_unified_frame_benchmark_v1/` |
| SERV-CT S2M2 fine-tuning | `01_frame_stereo/SERVCT/servct_s2m2_honest_finetune_gpu/` |
| SCARED S2M2 size/resolution tradeoff | `01_frame_stereo/SCARED/s2m2_size_tradeoff/`, `01_frame_stereo/SCARED/s2m2_resolution_tradeoff/` |
| SCARED transfer checks | `01_frame_stereo/SCARED/scared_s2m2_servct_transfer_dataset8_rectified/` |
| StereoAnyVideo temporal evaluation | `02_video_stereo/stereoanyvideo_temporal_eval/` |
| Video-stereo repo scouting | `02_video_stereo/video_stereo_repos/` |
| Temporal-refinement design | `03_temporal_refinement/design/argos_temporal_refinement_design/` |
| Temporal-refinement full-frame evaluation | `03_temporal_refinement/evaluation/temporal_refinement_evaluation_l736_v1/` |
| Temporal-refinement training runs | `03_temporal_refinement/training/` |
| Temporal-refinement caches | `03_temporal_refinement/cache/temporal_refinement_cache/` |
| Long SCARED sequence data/predictions | `04_dataset_derivatives/SCARED/scared_long_sequences/`, `04_dataset_derivatives/SCARED/scared_long_predictions/` |

## Frame Stereo

```text
01_frame_stereo/
  SERVCT/
    servct_unified_frame_benchmark_v1/
    servct_s2m2_honest_finetune_gpu/
    metrics/
    images/
  SCARED/
    s2m2_size_tradeoff/
    s2m2_resolution_tradeoff/
    scared_* audits and transfer checks
```

Use this section for geometric metrics with GT. Keep SERV-CT and SCARED grouped separately because protocols differ.

## Video Stereo

```text
02_video_stereo/
  stereoanyvideo_temporal_eval/
  video_stereo_repos/
```

Use this section for video-native stereo model scouting and temporal comparison against frame-based baselines.

## Temporal Refinement

```text
03_temporal_refinement/
  design/
  evaluation/
  debug/
  training/
    unet/
    convgru/
  cache/
```

Use `evaluation/temporal_refinement_evaluation_l736_v1/` for the current unified full-frame temporal comparison.

Training folders are development history. They are useful for understanding model evolution, but do not mix crop-validation training metrics with full-frame evaluation metrics.

## Dataset Artifacts

```text
04_dataset_derivatives/
  SCARED/
  logs/
```

This area contains generated artifacts derived from datasets, such as long extracted SCARED sequences, frozen prediction streams, and audit CSVs/logs. Raw datasets live under `dataset/`.

## Practical Rule

When making claims:

1. Use `01_frame_stereo/SERVCT/servct_unified_frame_benchmark_v1/` for SERV-CT frame-model numbers.
2. Use `01_frame_stereo/SCARED/s2m2_*tradeoff/` for SCARED S2M2 deployment tradeoffs.
3. Use `02_video_stereo/stereoanyvideo_temporal_eval/` for StereoAnyVideo vs frame-based temporal comparison.
4. Use `03_temporal_refinement/evaluation/temporal_refinement_evaluation_l736_v1/` for final temporal-refinement comparisons.
5. Use `../presentation/argos_progress/` for slide-ready material.
