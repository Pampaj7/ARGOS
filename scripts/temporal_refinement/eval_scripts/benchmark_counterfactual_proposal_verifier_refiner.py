#!/usr/bin/env python3
"""Benchmark Counterfactual Proposal Verifier refiner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

from benchmark_modern_refiner_v4 import real_batch  # noqa: E402
from counterfactual_proposal_verifier_refiner import counterfactual_proposal_verifier_refiner  # noqa: E402
from train_tiny_refiner_v1_full_gt import write_csv  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=Path("results/03_temporal_refinement/evaluation/s2m2_gt_refiner_targets_full"))
    p.add_argument("--output-root", type=Path, default=Path("results/03_temporal_refinement/training/counterfactual_proposal_verifier_refiner"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--residual-scale", type=float, default=32.0)
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = counterfactual_proposal_verifier_refiner(16, args.residual_scale).to(device).eval()
    ckpt = args.checkpoint or (args.output_root / "checkpoints" / "best_pareto.pt")
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
    params = sum(p.numel() for p in model.parameters())
    x_real, _raw = real_batch(args.targets_root, 4, device)
    with torch.no_grad():
        identity_delta = float(model(x_real, args.residual_scale)[2].abs().max())
    x = torch.randn(args.batch, 16, 256, 320, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(10):
            model(x, args.residual_scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            model(x, args.residual_scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
    ms = 1000.0 * (time.perf_counter() - t0) / (50 * args.batch)
    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    row = {"model": "counterfactual_proposal_verifier_refiner", "params": params, "batch": args.batch, "ms_per_frame_fp32": round(ms, 4), "peak_vram_mb": round(peak, 1), "identity_or_loaded_delta_px": identity_delta}
    write_csv(args.output_root / "benchmark_table.csv", [row])
    (args.output_root / "benchmark_summary.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
