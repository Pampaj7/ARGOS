# Standing instructions for ARGOS v2 agents

1. Always call the project ARGOS v2.
2. Before planning, inspect targeted relevant material under `/dtu/p1/leopam/ARGOS/SOTA`.
3. Run `python scripts/verify_freeze.py` before every experiment.
4. Never modify the frozen core during an experiment.
5. Never use future frames.
6. Never compose long optical-flow chains.
7. Align every anchor directly from the current frame to its original source frame.
8. Never write fused outputs into long-term memory.
9. Long-term memory contains only independently generated raw stereo predictions and required raw provenance.
10. Never add backbone IDs.
11. Never add backbone-internal cost volumes, features, hidden states, or confidence heads to the canonical method.
12. Never add the spatial critic, scalar gate, hard-negative critic, or safety ensemble to the canonical geometry method.
13. Safety metrics are diagnostic unless a new safety protocol is separately proposed, validated, and frozen.
14. Dataset IDs: training 1, 3, 6; validation/calibration 2; test 7.
15. Dataset 7 is held out from the reported frozen training and validation protocol but is not a project-wide pristine test set.
16. Freeze all choices before any test evaluation.
17. Follow PONYTAIL and YAGNI.
18. Avoid dense prediction caches by default.
19. Save compact CSV, JSON, README, logs, and small diagnostic contact sheets.
20. Keep dataset preparation, model inference, and benchmarking separated.
21. Reuse validated caches and inference paths.
22. Do not overwrite validated results.
23. Use the available 2 x H100 PCIe 80 GB, AMD EPYC CPUs, and approximately 1 TB RAM efficiently.
24. Use many DataLoader workers where useful.
25. Do not artificially limit workers on full runs.
26. Keep smoke tests small.
27. Delete successful temporary smoke-test outputs.
28. Detach long runs only after PID, GPU, log-growth, first-batch/sequence, and memory checks.
29. Never use `git clean`, `git reset --hard`, or broad `git restore`.
30. Do not modify unrelated files.

All future experiments live under `/dtu/p1/leopam/ARGOS/ARGOS_FREEZED/experiments/`.

## Multi-agent orchestration for ARGOS v2

- The parent model is the orchestrator and retains final responsibility.
- Use `scout_luna` for repository and experiment exploration.
- Use `tester_luna` for smoke tests, logs, builds, and validation.
- Use `worker_terra` only for bounded experiment-local implementation.
- Use `reviewer_terra` after meaningful changes.
- Use `architect_sol` for architecture or scientific-protocol escalation.
- Never modify the frozen `geometry_v1` core, hashes, checkpoints, or baselines.
- All experiment code and outputs belong under `/dtu/p1/leopam/ARGOS/ARGOS_FREEZED/experiments/`.
- Inspect targeted material under `/dtu/p1/leopam/ARGOS/SOTA/` before research planning.
- Follow PONYTAIL and YAGNI; avoid broad recursive `find` on DTU storage and destructive Git commands.
- One writer per worktree; read-only roles must remain read-only.
- Training runs use the available H100s and AMD EPYC workers efficiently; smoke tests stay small and successful temporary outputs are deleted.
- Detach long runs only after PID, GPU, log-growth, first-batch/sequence, and memory health checks.
- Dataset 7 remains closed until the relevant experiment recipe is frozen.
- Never let a subagent alter frozen hashes or architecture files.
