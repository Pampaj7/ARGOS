#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from scripts.argos_paths import RESULTS_DIR
from scripts.temporal_refinement.playground.runner import (
    run_all_smokes,
    run_config_smoke,
    run_gt_short_race,
    run_real_gpu_smokes,
    run_short_race,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARGOS modular temporal-refinement playground.")
    parser.add_argument(
        "--mode",
        choices=["synthetic-smoke", "real-gpu-smoke", "short-race", "gt-short-race"],
        default="synthetic-smoke",
    )
    parser.add_argument("--config", type=Path, help="Single experiment YAML for synthetic smoke mode.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/temporal_refinement/playground"))
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "03_temporal_refinement/playground/tmp")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sequence-id", default="test_dataset_9_keyframe_1")
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--crop-width", type=int, default=384)
    parser.add_argument("--raft-iters", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--updates-per-epoch", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
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

    if args.mode == "short-race":
        payload = run_short_race(
            config_dir=args.config_dir,
            output_dir=args.out_dir,
            epochs=args.epochs,
            updates_per_epoch=args.updates_per_epoch,
            sequence_length=args.sequence_length,
            crop_height=args.crop_height,
            crop_width=args.crop_width,
            val_sequence_id=args.sequence_id,
            raft_iters=args.raft_iters,
            seed=args.seed,
        )
        print(f"Ran short race for {payload['active_models']} -> {args.out_dir}")
        return

    if args.mode == "gt-short-race":
        payload = run_gt_short_race(
            config_dir=args.config_dir,
            output_dir=args.out_dir,
            epochs=args.epochs,
            updates_per_epoch=args.updates_per_epoch,
            sequence_length=args.sequence_length,
            crop_height=args.crop_height,
            crop_width=args.crop_width,
            raft_iters=args.raft_iters,
            seed=args.seed,
        )
        print(f"Ran GT short race -> {payload['output_dir']}")
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
