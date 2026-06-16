# ARGOS Scripts

This folder tracks ARGOS-specific scripts and adapters. Upstream model repositories are not vendored here; they live locally under `/home/pampaj/Desktop/stereo/` and are ignored by the ARGOS git repo.

The script layer is meant to make the project paper-reproducible while keeping the repository lightweight:

- converters define the shared surgical dataset format;
- evaluators adapt upstream stereo repos to the same SERV-CT metrics;
- report scripts regenerate scoreboards and paper figures;
- download scripts document background dataset/model queues.

External upstream repos live under `/home/pampaj/Desktop/stereo/`; ARGOS keeps
the adapters and report builders here so the paper-facing workflow is easier to
reproduce.

## ARGOS-Native Utilities

- `converters/convert_servct_to_argos.py`: converts SERV-CT into the local unified ARGOS sample format.
- `converters/convert_scared_to_argos.py`: placeholder/status tool until SCARED download/extraction is complete.
- `run_all_servct_baselines.py`: lightweight wrapper for regenerating current SERV-CT reports.
- `reports/make_servct_scoreboard.py`: creates the current SERV-CT scoreboard CSV/MD/PNG.
- `reports/build_servct_unified_benchmark.py`: builds the canonical SERV-CT accuracy benchmark package.
- `reports/run_servct_runtime_benchmark.py`: adds local adapter-level runtime/VRAM measurements to the SERV-CT package.
- `reports/build_presentation_assets.py`: builds the Monday presentation asset package.
- `defom_stereo/eval_servct_defom.py`: SERV-CT evaluator for DEFOM-Stereo checkpoints.
- `temporal_refinement/train_refiner.py`: stable unified entrypoint for Tiny U-Net and causal ConvGRU temporal refiner training.
- `temporal_refinement/train_temporal_refiner_fastcache.py`: implementation of the indexed fast-cache trainer.
- `temporal_refinement/evaluate_temporal_refinement.py`: unified full-frame temporal-refinement evaluation.

## Temporal Refinement Scripts

Current entrypoints:

- `temporal_refinement/train_refiner.py`: preferred CLI wrapper.
- `temporal_refinement/train_temporal_refiner_fastcache.py`: trainer implementation.
- `temporal_refinement/evaluate_temporal_refinement.py`: full-frame evaluation.
- `temporal_refinement/build_large_v3_s2m2s512_fast_cache.py`: indexed fast-cache builder.
- `temporal_refinement/extract_scared_long_sequences.py`: long SCARED sequence extraction.
- `temporal_refinement/predict_s2m2_long_sequences.py`: frozen S2M2 predictions for long sequences.
- `temporal_refinement/predict_stereoanyvideo_long_sequences.py`: frozen StereoAnyVideo teacher predictions.

Legacy debug trainers are kept under `temporal_refinement/legacy/` for history.

## Current Script Priorities

- Keep SERV-CT evaluators aligned on the same metrics and output schema.
- Extend `convert_scared_to_argos.py` after full SCARED extraction.
- Add a real `run_all_servct_baselines.py` command list for repeatable baseline regeneration.
- Add robustness analysis scripts for near-field, boundaries, and specular/textureless regions.
- Compare Tiny U-Net window refinement against causal ConvGRU refinement on the same fast indexed cache.
- Keep all generated datasets, checkpoints, upstream repos, and bulky logs out of git.
