# Scripts

This folder is for ARGOS-specific scripts and adapters. Upstream model repositories are not vendored here.

Current source scripts live in the local model workspaces while experiments are still moving quickly:

- `/home/pampaj/Desktop/stereo/s2m2/scripts/finetune_servct_s2m2.py`
- `/home/pampaj/Desktop/stereo/s2m2/scripts/eval_servct_s2m2.py`
- `/home/pampaj/Desktop/stereo/Fast-FoundationStereo/scripts/eval_servct_onnx.py`
- `/home/pampaj/Desktop/stereo/stereoanywhere/scripts/eval_servct_stereoanywhere.py`
- `/home/pampaj/Desktop/stereo/MonSter-plusplus/RT-MonSter++/scripts_eval_servct_monster.py`
- `/home/pampaj/Desktop/stereo/download_jobs/download_monsterpp_large.py`

New ARGOS-native utilities:

- `converters/convert_servct_to_argos.py`: converts SERV-CT into the local unified ARGOS sample format.
- `converters/convert_scared_to_argos.py`: placeholder/status tool until SCARED download/extraction is complete.
- `run_all_servct_baselines.py`: lightweight wrapper for regenerating current SERV-CT reports.
- `reports/make_servct_scoreboard.py`: creates the current SERV-CT scoreboard CSV/MD/PNG.
- `defom_stereo/eval_servct_defom.py`: SERV-CT evaluator for DEFOM-Stereo checkpoints.

Stable versions can be copied here as the project settles.
