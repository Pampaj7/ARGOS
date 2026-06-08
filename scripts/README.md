# ARGOS Scripts

This folder tracks ARGOS-specific scripts and adapters. Upstream model repositories are not vendored here; they live locally under `/home/pampaj/Desktop/stereo/` and are ignored by the ARGOS git repo.

The script layer is meant to make the project paper-reproducible while keeping the repository lightweight:

- converters define the shared surgical dataset format;
- evaluators adapt upstream stereo repos to the same SERV-CT metrics;
- report scripts regenerate scoreboards and paper figures;
- download scripts document background dataset/model queues.

Current source scripts live in the local model workspaces while experiments are still moving quickly:

- `stereo/s2m2/scripts/finetune_servct_s2m2.py`
- `stereo/s2m2/scripts/eval_servct_s2m2.py`
- `stereo/Fast-FoundationStereo/scripts/eval_servct_onnx.py`
- `stereo/stereoanywhere/scripts/eval_servct_stereoanywhere.py`
- `stereo/MonSter-plusplus/RT-MonSter++/scripts_eval_servct_monster.py`
- `stereo/download_jobs/download_monsterpp_large.py`

## ARGOS-Native Utilities

- `converters/convert_servct_to_argos.py`: converts SERV-CT into the local unified ARGOS sample format.
- `converters/convert_scared_to_argos.py`: placeholder/status tool until SCARED download/extraction is complete.
- `run_all_servct_baselines.py`: lightweight wrapper for regenerating current SERV-CT reports.
- `reports/make_servct_scoreboard.py`: creates the current SERV-CT scoreboard CSV/MD/PNG.
- `defom_stereo/eval_servct_defom.py`: SERV-CT evaluator for DEFOM-Stereo checkpoints.

## Current Script Priorities

- Keep SERV-CT evaluators aligned on the same metrics and output schema.
- Extend `convert_scared_to_argos.py` after full SCARED extraction.
- Add a real `run_all_servct_baselines.py` command list for repeatable baseline regeneration.
- Add robustness analysis scripts for near-field, boundaries, and specular/textureless regions.
- Keep all generated datasets, checkpoints, upstream repos, and bulky logs out of git.
