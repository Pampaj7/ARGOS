#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from scripts.argos_paths import RESULTS_DIR
from scripts.temporal_refinement.playground.runner import run_all_smokes, run_config_smoke, run_real_gpu_smokes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARGOS modular temporal-refinement playground smoke tests.")
    parser.add_argument("--mode", choices=["synthetic-smoke", "real-gpu-smoke"], default="synthetic-smoke")
    parser.add_argument("--config", type=Path, help="Single experiment YAML.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/temporal_refinement/playground"))
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "03_temporal_refinement/playground/tmp_smoke")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sequence-id", default="test_dataset_9_keyframe_1")
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--crop-width", type=int, default=384)
    parser.add_argument("--raft-iters", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "real-gpu-smoke":
        rows = run_real_gpu_smokes(
            config_dir=args.config_dir,
            output_dir=args.out_dir,
            sequence_id=args.sequence_id,
            sequence_length=args.sequence_length,
            crop_height=args.crop_height,
            crop_width=args.crop_width,
            raft_iters=args.raft_iters,
        )
        print(f"Ran {len(rows)} real GPU playground smoke configs -> {args.out_dir}")
        return

    device = torch.device(args.device)
    if args.config:
        metrics = run_config_smoke(args.config, args.out_dir / args.config.stem, device)
        print(metrics)
    else:
        rows = run_all_smokes(args.config_dir, args.out_dir, device)
        print(f"Ran {len(rows)} playground smoke configs -> {args.out_dir}")


if __name__ == "__main__":
    main()
