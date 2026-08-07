# ARGOS v2 — engineering review B

Reviewed before full training on 2026-08-05.

CRITICAL ISSUES: 0

- Frozen-core, checkpoint, SEA-RAFT, experiment-source, and seed-initialization hashes are verified before launcher and run entry points proceed.
- Checkpoints and JSON/CSV state are written through same-filesystem temporary files followed by atomic replacement.
- Resume validates seed, budget, scheduler horizon, steps/epoch, initialization hash, and all recipe hashes before loading optimizer/scheduler/scaler state.
- Outputs resolve under the registered experiment directories; full runs reject alternate destinations and truncated inputs.
- Two disjoint one-GPU LSF slots run one process per GPU, propagate non-zero exits, record PID/job/host/GPU slot/log, never retry blindly, and leave failed logs intact.
- A kernel-released `flock` serializes each slot across interactive use and its queued batch takeover, preventing duplicate writes while allowing checkpoint resume after job loss.
- Only best and final recovery checkpoints are retained. Evidence banks are ephemeral RAM objects; no dense evidence or prediction caches are persisted.
- DataLoader workers are canonical (48, persistent, prefetch 4); two-run host-memory estimate is 180 GiB against the observed uncapped cgroup and approximately 1 TiB node RAM.
- Multiprocessing temporary directories are forced onto node-local `/tmp`; the smoke-discovered NFS finalizer issue is resolved without changing numerical work.
- Full campaign launch is delegated to LSF so it does not depend on the interactive SSH shell.
- Compact atomic `state.json` supports monitoring without rescanning full logs.
- The dataset guard is fail-closed in all train/validation data loaders; final-test execution remains independently locked.
- The smoke-discovered bare integer-ID gap was fixed in the shared guard; integer `7`, `dataset_id=7`, and `dataset_7` paths all fail before loading.

Result: PASS for smoke and full-launch engineering.
