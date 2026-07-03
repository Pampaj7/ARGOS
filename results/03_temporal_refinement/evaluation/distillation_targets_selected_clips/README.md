# Selected-Clip Distillation Targets

This run generated compact teacher/oracle targets for selected planning clips only.
Oracle labels use ground truth, so they are training targets and are not deployable online.

- Planning CSV: `/dtu/p1/leopam/ARGOS/results/03_temporal_refinement/evaluation/distillation_planning/candidate_clips_for_distillation.csv`
- Dataset root: `/dtu/p1/leopam/ARGOS/dataset/SCARED/curated/temporal_gt_rectified`
- Clips processed: `6`
- Target scale: `0.25`
- Downsample min valid ratio: `0.25`
- Full-resolution targets: `False`
- Candidate caches: not written. Dense candidate predictions live only in memory.
- Targets include raw S2M2, fixed EMA, adaptive no-RAFT, RAFT-Small, and SAV when available.
- SAV chunking: chunk size `32`, pad last chunk `True`.
- `oracle_all_available` includes SAV when `teacher_availability_report.json` marks SAV available.
- RAFT-Small and StereoAnyVideo: offline teacher candidates only; see `teacher_availability_report.json`.
- These outputs are selected-clip targets for future lightweight distillation.
- Saved `.npz` metrics are low-resolution target-space metrics, computed after valid-mask-aware downsampling and low-resolution oracle selection.

Targets are stored as compressed `.npz` files under `clips/*/targets/`. Full-resolution target export requires explicit `--save-full-resolution-targets true`.
