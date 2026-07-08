#!/usr/bin/env python3
"""Matrix driver: load the SCARED split shards (raw/gt/valid + cached rgb/warp_flow/occ) ONCE,
then train a list of (config, seed) runs reusing the in-RAM shards. This avoids re-decompressing
the ~8 GB aux cache per run (the dominant per-run cost). Run two of these in parallel (one per
H100, over disjoint config halves) to use both GPUs. Resumable: skips runs whose config.json exists.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_nvds_lite as TR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", required=True, help="config letters, e.g. A B C")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=1200)
    ap.add_argument("--out", type=Path, default=TR.OUT / "runs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    t0 = time.time()
    train = TR.load_split_shards("train")
    val = TR.load_split_shards("val")
    test = TR.load_split_shards("test")
    print(f"[driver] loaded shards once in {time.time() - t0:.1f}s; device={device}", flush=True)

    for cfg in args.configs:
        for seed in args.seeds:
            rid = TR.run_id_for(cfg, args.clip_len, seed)
            if (args.out / rid / "config.json").exists():
                print(f"[driver] skip {rid}", flush=True)
                continue
            run_args = SimpleNamespace(config=cfg, clip_len=args.clip_len, steps=args.steps,
                                       batch=args.batch, lr=args.lr, seed=seed,
                                       eval_every=args.eval_every, out=args.out,
                                       device=args.device, smoke=False)
            t1 = time.time()
            TR.train_run(run_args, train, val, test, device)
            print(f"[driver] done {rid} in {time.time() - t1:.1f}s", flush=True)
    print("RUN_MATRIX_DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
