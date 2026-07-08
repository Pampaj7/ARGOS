#!/usr/bin/env python3
"""Evaluate a trained Counterfactual Proposal Verifier checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

from counterfactual_proposal_verifier_refiner import counterfactual_proposal_verifier_refiner  # noqa: E402
from train_counterfactual_proposal_verifier_refiner import full_gt_eval_cpv, selected_eval  # noqa: E402
from train_tiny_refiner_v1_full_gt import load_shards, write_csv  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import FullFrameDataset, load_samples_with_split, write_csv_union  # noqa: E402
from train_tiny_refiner_v3_2_hybrid_oracle import load_clips, make_loader  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path("results/03_temporal_refinement/training/counterfactual_proposal_verifier_refiner"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    ckpt = args.checkpoint or (args.output_root / "checkpoints" / "best_pareto.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = argparse.Namespace(**ck["args"])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = counterfactual_proposal_verifier_refiner(ck.get("input_channels", 16), cfg.residual_scale).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    _splits, by_split = load_samples_with_split(cfg.targets_root, cfg.balanced_split_json, cfg.max_frames)
    shards = load_shards(by_split["train"] + by_split["val"] + by_split["test"])
    clips = load_clips(cfg.oracle_targets_root, cfg)
    full_ds = {split: FullFrameDataset(samples, shards, cfg.context_frames) for split, samples in by_split.items()}
    loaders = {split: make_loader(ds, cfg.eval_batch_size, max(0, cfg.num_workers // 2), False, cfg.prefetch_factor) for split, ds in full_ds.items()}
    sel, frames, _pred = selected_eval(model, clips, cfg, device)
    fg = {split: full_gt_eval_cpv(model, loaders[split], device, cfg) for split in ("val", "test")}
    write_csv_union(args.output_root / "selected_oracle_metrics.csv", frames)
    write_csv_union(args.output_root / "selected_oracle_frame_metrics.csv", frames)
    write_csv(args.output_root / "full_gt_val_metrics.csv", [fg["val"]])
    write_csv(args.output_root / "full_gt_test_metrics.csv", [fg["test"]])
    summary = {"checkpoint": str(ckpt), "selected_all": sel["all"], "selected_pathological": sel["patho"], "selected_clean": sel["clean"], "full_gt_val": fg["val"], "full_gt_test": fg["test"]}
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
