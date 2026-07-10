#!/usr/bin/env python3
"""Full 17-sequence x 5-backbone SCARED-C cache generation (85 jobs). LPT-scheduled
(longest jobs first) into a shared queue across the 2 allocated H100 GPUs, so a GPU that
finishes early immediately grabs the next-largest remaining job instead of idling near the
end — the pilot's plain-FIFO queue let one GPU idle for a while as the other finished a
single very long StereoAnywhere job. Every job is independently resumable
(run_backbone_cache.py self-skips if its cache already re-validates), and failed jobs are
retried once automatically.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from argos_v2.backbones import BACKBONE_NAMES
from argos_v2.paths import QUALITY_GATE_CSV, RESULTS_DIR, V2_ROOT
from argos_v2.sequences import accepted_sequences

N_GPUS = 2

# Rough per-frame duration estimates from the pilot's timing_summary.csv — used only to
# order the job queue (longest first); not treated as ground truth anywhere else.
RATE_S_PER_FRAME = {
    "StereoAnywhere": 1.3, "RAFT-Stereo": 0.55, "CREStereo": 0.18,
    "S2M2-S": 0.085, "Fast-FoundationStereo": 0.026,
}

LOG_DIR = RESULTS_DIR / "full_run/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG = RESULTS_DIR / "full_run/run_full.jsonl"


def sequence_frame_counts() -> dict[str, int]:
    counts = {}
    with QUALITY_GATE_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["status"] == "pass":
                counts[r["sequence_id"]] = int(r["included_count_full"])
    return counts


def build_job_queue() -> list[tuple[str, str, float]]:
    counts = sequence_frame_counts()
    seqs = accepted_sequences()
    jobs = [(backbone, seq, RATE_S_PER_FRAME[backbone] * counts[seq]) for backbone in BACKBONE_NAMES for seq in seqs]
    jobs.sort(key=lambda j: j[2], reverse=True)  # LPT: longest estimated job first
    return jobs


def run_job(gpu_id: int, backbone: str, sequence: str, attempt: int, max_frames: int = 0) -> dict:
    log_path = LOG_DIR / f"{backbone}_{sequence}_attempt{attempt}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "run_backbone_cache.py"),
        "--backbone", backbone, "--sequence", sequence,
    ]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    t0 = time.perf_counter()
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, env=env, cwd=str(V2_ROOT), stdout=logf, stderr=subprocess.STDOUT)
    wall_s = time.perf_counter() - t0
    result = {
        "backbone": backbone, "sequence": sequence, "gpu": gpu_id, "attempt": attempt,
        "wall_s": wall_s, "returncode": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        "log": str(log_path), "timestamp": time.time(),
    }
    print(f"[gpu{gpu_id}] {backbone}/{sequence} (attempt {attempt}): {result['status']} in {wall_s:.1f}s", flush=True)
    with RUN_LOG.open("a") as f:
        f.write(json.dumps(result) + "\n")
    return result


def worker(gpu_id: int, job_q: "queue.Queue", results: list, lock: threading.Lock, retry_q: "queue.Queue", max_frames: int):
    while True:
        try:
            backbone, sequence, _est = job_q.get_nowait()
        except queue.Empty:
            return
        result = run_job(gpu_id, backbone, sequence, attempt=1, max_frames=max_frames)
        with lock:
            results.append(result)
        if result["returncode"] != 0:
            retry_q.put((backbone, sequence))
        job_q.task_done()


def retry_worker(gpu_id: int, retry_q: "queue.Queue", results: list, lock: threading.Lock, max_frames: int):
    while True:
        try:
            backbone, sequence = retry_q.get_nowait()
        except queue.Empty:
            return
        result = run_job(gpu_id, backbone, sequence, attempt=2, max_frames=max_frames)
        with lock:
            results.append(result)
        retry_q.task_done()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only run the first N (post-LPT-sort) jobs")
    ap.add_argument("--max-frames", type=int, default=0, help="smoke test: cap frames per job (writes to _smoke_ namespace)")
    args = ap.parse_args()

    jobs = build_job_queue()
    seqs = accepted_sequences()
    expected = len(BACKBONE_NAMES) * len(seqs)
    print(f"full run: {len(jobs)} jobs ({len(BACKBONE_NAMES)} backbones x {len(seqs)} sequences = {expected})", flush=True)
    assert len(jobs) == expected
    if args.limit:
        jobs = jobs[: args.limit]
        print(f"--limit {args.limit}: smoke-testing only {len(jobs)} jobs", flush=True)

    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    if RUN_LOG.exists():
        RUN_LOG.unlink()  # fresh append-log per invocation; individual job caches remain resumable regardless

    job_q: "queue.Queue" = queue.Queue()
    for j in jobs:
        job_q.put(j)
    retry_q: "queue.Queue" = queue.Queue()
    results: list[dict] = []
    lock = threading.Lock()

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(gpu, job_q, results, lock, retry_q, args.max_frames)) for gpu in range(N_GPUS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not retry_q.empty():
        n_retry = retry_q.qsize()
        print(f"\nretrying {n_retry} failed jobs once...", flush=True)
        rthreads = [threading.Thread(target=retry_worker, args=(gpu, retry_q, results, lock, args.max_frames)) for gpu in range(N_GPUS)]
        for t in rthreads:
            t.start()
        for t in rthreads:
            t.join()

    total_s = time.perf_counter() - t0
    final = {}
    for r in results:
        key = (r["backbone"], r["sequence"])
        if key not in final or r["attempt"] > final[key]["attempt"]:
            final[key] = r
    n_failed = sum(1 for r in final.values() if r["returncode"] != 0)
    print(f"\nfull run done: {len(final)}/{len(jobs)} jobs, {n_failed} still failed after retry, {total_s:.1f}s wall", flush=True)
    if n_failed:
        for r in final.values():
            if r["returncode"] != 0:
                print(f"  STILL FAILING: {r['backbone']}/{r['sequence']} (see {r['log']})", flush=True)
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
