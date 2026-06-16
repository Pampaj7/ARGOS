# ARGOS Repository Map

This file is the quick orientation map for the local ARGOS workspace.

## What Matters First

| Path | Purpose | Notes |
|---|---|---|
| `README.md` | project narrative and current scientific direction | broad ARGOS-Wound overview |
| `docs/STATUS.md` | chronological status log | update when a result changes the story |
| `docs/EXPERIMENT_PROTOCOL.md` | split and evaluation rules | especially SERV-CT honest vs all-data adaptation |
| `docs/DATASETS.md` | dataset inventory and formats | raw data and converted ARGOS data |
| `docs/MODEL_ZOO.md` | model/repo status | what is runnable, blocked, or pending |
| `results/README.md` | result directory index | start here for tables/plots |
| `scripts/README.md` | script entrypoint index | start here before launching experiments |

## Code

| Path | Purpose |
|---|---|
| `scripts/temporal_refinement/lib/` | temporal refiner library: datasets, models, losses, metrics, training helpers |
| `scripts/temporal_refinement/train_refiner.py` | stable unified trainer entrypoint |
| `scripts/temporal_refinement/train_temporal_refiner_fastcache.py` | full fast-cache trainer implementation |
| `scripts/temporal_refinement/evaluate_temporal_refinement.py` | unified full-frame temporal evaluation |
| `scripts/reports/` | report/table/plot generation scripts |
| `scripts/s2m2/`, `scripts/raft_stereo/`, etc. | SERV-CT adapters for external stereo repos |
| `tests/` | focused tests for ConvGRU/refiner/evaluation code |

## Data

| Path | Purpose | Size Class |
|---|---|---|
| `dataset/SCARED/` | SCARED raw archives, curated clips, and workspace extracts | huge, ignored |
| `dataset/SERVCT/` | SERV-CT raw archive plus ARGOS-format GT samples | medium, ignored |
| `dataset/StereoMIS/` | downloaded StereoMIS archive, inventory, metadata, preview | large, ignored |
| `dataset/D4D/` | D4D metadata and staged specimen downloads | large, ignored |
| `dataset/EndoSLAM/` | EndoSLAM support data | medium, ignored |
| `results/04_dataset_derivatives/SCARED/scared_long_sequences/` | extracted long SCARED frame streams | large |
| `results/04_dataset_derivatives/SCARED/scared_long_predictions/` | frozen S2M2/StereoAnyVideo predictions | large |
| `results/03_temporal_refinement/cache/temporal_refinement_cache/` | `.npz`/indexed training caches | very large, ignored payload |

## Results To Use In Slides

| Path | What It Contains |
|---|---|
| `results/01_frame_stereo/SERVCT/servct_unified_frame_benchmark_v1/` | main SERV-CT frame-stereo table, runtime/VRAM addendum, plots |
| `presentation/argos_progress/` | Monday presentation tables, plots, diagrams, videos, GIFs |
| `results/servct evaluation/` | simple SERV-CT model baseline table |
| `results/03_temporal_refinement/evaluation/temporal_refinement_evaluation_l736_v1/` | unified full-frame temporal-refinement comparison |
| `results/02_video_stereo/stereoanyvideo_temporal_eval/` | StereoAnyVideo vs S2M2 temporal evaluation |
| `results/01_frame_stereo/SCARED/s2m2_size_tradeoff/` and `results/01_frame_stereo/SCARED/s2m2_resolution_tradeoff/` | S2M2 model-size/resolution tradeoffs |

## Results To Treat As Development History

| Path Pattern | Notes |
|---|---|
| `results/temporal_refinement_debug_unet_*` | tiny debug/overfit runs; useful for method history |
| `results/temporal_refinement_train_unet_*` | U-Net training experiments before ConvGRU became the main path |
| `results/temporal_refinement_train_convgru_*` | ConvGRU probes and long runs |
| `results/scared_*_audit/` | dataset/model diagnostics |

## External Repos

Downloaded upstream stereo repos live outside this ARGOS repo under:

`/home/pampaj/Desktop/stereo/`

ARGOS tracks adapters, reports, configs, and compact evidence. It does not
vendored upstream repos or model weights.

## Naming Rules Going Forward

- New final-ish result folders should use a clear prefix:
  - `servct_*`
  - `scared_*`
  - `temporal_refinement_*`
  - `presentation_*`
- Every result folder should have a `README.md`.
- Big payloads should stay ignored; save compact `csv/json/md/png` summaries.
- For SERV-CT, always label:
  - `zero-shot`
  - `honest fine-tune`
  - `all-data/all-surgical upper bound`
