#!/usr/bin/env python3
"""Pilot orchestrator: 5 backbones x 2 SCARED-C sequences = 10 jobs, dispatched across the
2 allocated H100 GPUs via a shared queue (not a static split) so a GPU that finishes early
immediately picks up the next job instead of idling.

Each job runs run_backbone_cache.py as its own OS subprocess (import isolation + per-job
CUDA_VISIBLE_DEVICES pinning).
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from argos_v2.backbones import BACKBONE_NAMES
from argos_v2.paths import RESULTS_DIR, V2_ROOT

PILOT_SEQUENCES = ["dataset_3_keyframe_1", "dataset_7_keyframe_4"]  # small (329) + large (2197)
N_GPUS = 2

LOG_DIR = RESULTS_DIR / "pilot/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_job(gpu_id: int, backbone: str, sequence: str) -> dict:
    log_path = LOG_DIR / f"{backbone}_{sequence}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "run_backbone_cache.py"),
        "--backbone", backbone, "--sequence", sequence,
    ]
    t0 = time.perf_counter()
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, env=env, cwd=str(V2_ROOT), stdout=logf, stderr=subprocess.STDOUT)
    wall_s = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"[gpu{gpu_id}] {backbone}/{sequence}: {status} in {wall_s:.1f}s", flush=True)
    return {"backbone": backbone, "sequence": sequence, "gpu": gpu_id, "wall_s": wall_s, "returncode": proc.returncode}


def worker(gpu_id: int, job_q: "queue.Queue", results: list, lock: threading.Lock):
    while True:
        try:
            backbone, sequence = job_q.get_nowait()
        except queue.Empty:
            return
        result = run_job(gpu_id, backbone, sequence)
        with lock:
            results.append(result)
        job_q.task_done()


def main() -> int:
    jobs = [(b, s) for b in BACKBONE_NAMES for s in PILOT_SEQUENCES]
    job_q: "queue.Queue" = queue.Queue()
    for j in jobs:
        job_q.put(j)

    results: list[dict] = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(gpu, job_q, results, lock)) for gpu in range(N_GPUS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_s = time.perf_counter() - t0

    n_failed = sum(1 for r in results if r["returncode"] != 0)
    print(f"\npilot done: {len(results)}/{len(jobs)} jobs, {n_failed} failed, {total_s:.1f}s wall", flush=True)
    for r in results:
        print(r, flush=True)
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
