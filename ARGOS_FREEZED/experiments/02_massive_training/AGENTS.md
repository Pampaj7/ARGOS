# ARGOS v2 campaign constraints

- Run `/dtu/p1/leopam/ARGOS/ARGOS_FREEZED/scripts/verify_freeze.py` before every action.
- Modify files only within this experiment or `01_multiseed_replication`.
- Architecture, features, losses, optimizer, batch size, sampler, thresholds and splits are frozen.
- Only optimization budget may vary.
- Training/validation must reject dataset 7 and paths containing `dataset_7`.
- Never import ARGOS-V2 from full training processes.
- Never use a critic, safety gate, recurrent corrected state, fused writeback, future frame, or composed flow.
- Preserve failed logs; resume only after root-cause diagnosis.
- Follow PONYTAIL and YAGNI.
