#!/usr/bin/env python3
"""Full pilot matrix runner (resumable: skips runs whose config.json exists)."""
import itertools, json, subprocess, sys
from pathlib import Path
ROOT = Path("/dtu/p1/leopam/ARGOS")
RUNS = ROOT / "results/03_temporal_refinement/adaptation/d4d_few_shot_pilot/runs"
PY = str(ROOT / ".miniconda/envs/argos/bin/python")
TRAINER = str(ROOT / "scripts/temporal_refinement/adaptation/train_d4d_few_shot_adapter.py")
MODELS = ["v3.2c", "EGBM-v3-CARE-S"]
SIZES = ["1session", "2session", "4session", "8session"]
SEEDS = [0, 1, 2]
jobs = []
for m in MODELS:
    jobs.append((m, "zero_shot", "none", 0))
    for mode, size, seed in itertools.product(["calibration_only", "head_only", "full"], SIZES, SEEDS):
        jobs.append((m, mode, f"{size}_seed{seed}", seed))
    for size, seed in itertools.product(["4session", "8session"], SEEDS):
        jobs.append((m, "scratch", f"{size}_seed{seed}", seed))
skip_empty = {"1session_seed1"}  # 0 usable anchors
done = failed = skipped = 0
for m, mode, split, seed in jobs:
    rid = f"{m.replace('.','_')}__{mode}__{split}__seed{seed}"
    if split in skip_empty:
        skipped += 1; continue
    if (RUNS / rid / "config.json").exists():
        done += 1; continue
    r = subprocess.run([PY, TRAINER, "--model", m, "--adaptation-mode", mode,
                        "--split", split, "--seed", str(seed)],
                       capture_output=True, text=True)
    if r.returncode == 0:
        done += 1
        print(f"[ok] {rid}", flush=True)
    else:
        failed += 1
        print(f"[FAIL] {rid}: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}", flush=True)
print(json.dumps({"done": done, "failed": failed, "skipped_empty": skipped, "total": len(jobs)}))
