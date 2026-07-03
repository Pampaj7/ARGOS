#!/usr/bin/env python3
"""Benchmark the experimental EGBM-Refiner against v3 and the v4 prototypes.

Forward-only: params, ms/frame (fp32, batched + single), peak VRAM, identity-at-init on a
real full-GT frame, output/diagnostic shape checks. No training, no teacher inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

from train_tiny_refiner_v1_full_gt import DEFAULT_TARGETS_ROOT  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import AbstentionCropRefiner  # noqa: E402
from benchmark_modern_refiner_v4 import real_batch  # noqa: E402
from modern_refiner_v4 import v4_small, v4_tiny  # noqa: E402
from experimental_refiner_vx import egbm_refiner, egbm_refiner_large  # noqa: E402


DEFAULT_OUTPUT = Path("results/03_temporal_refinement/training/experimental_refiner_vx_benchmark")
REFERENCE = {
    "v3": {"params": 194818, "ms_per_frame": 1.0786},
    "v4_tiny": {"params": 942867, "ms_per_frame": 2.7402},
    "v4_small": {"params": 2527131, "ms_per_frame": 4.6701},
}


def forward3(model: torch.nn.Module, x: torch.Tensor, scale: float):
    return model(x, scale)[:3]


@torch.no_grad()
def time_model(model: torch.nn.Module, x: torch.Tensor, scale: float, device: torch.device, warmup: int = 10, iters: int = 50) -> tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    for _ in range(warmup):
        forward3(model, x, scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        forward3(model, x, scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = 1000.0 * (time.perf_counter() - t0) / (iters * x.shape[0])
    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    return ms, peak


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--in-channels", type=int, default=16)
    p.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 32])
    p.add_argument("--residual-scale", type=float, default=3.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    log = [f"device={device}", f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}"]

    models = {
        "v3_abstention_crop_refiner": AbstentionCropRefiner(args.in_channels),
        "v4_tiny": v4_tiny(args.in_channels, args.residual_scale),
        "v4_small": v4_small(args.in_channels, args.residual_scale),
        "egbm_refiner": egbm_refiner(args.in_channels, args.residual_scale),
        "egbm_refiner_large": egbm_refiner_large(args.in_channels, args.residual_scale),
    }
    rows: list[dict[str, Any]] = []
    sanity: dict[str, Any] = {}
    for name, model in models.items():
        model = model.to(device).eval()
        params = sum(q.numel() for q in model.parameters())
        x_real, raw = real_batch(args.targets_root, 4, device)
        with torch.no_grad():
            out = model(x_real, args.residual_scale)
        bad_logit, p_bad, residual = out[:3]
        refined = raw + residual if name.startswith("egbm") else raw + p_bad * residual
        entry: dict[str, Any] = {
            "params": params,
            "outputs_finite": bool(torch.isfinite(bad_logit).all() and torch.isfinite(p_bad).all() and torch.isfinite(residual).all()),
            "identity_at_init_max_abs_delta": float((refined - raw).abs().max()),
            "p_bad_at_init": float(p_bad.mean()),
            "output_shapes": {"bad_logit": list(bad_logit.shape), "residual": list(residual.shape)},
        }
        if name.startswith("egbm"):
            diag = out[3]
            entry["diagnostics"] = {k: list(v.shape) for k, v in diag.items()}
            entry["identity_router_weight_at_init"] = float(diag["router_weights"][:, -1].mean())
            entry["dynamic_threshold_at_init"] = float(diag["dynamic_threshold"].mean())
        sanity[name] = entry
        for batch in args.batch_sizes:
            x = torch.randn(batch, args.in_channels, args.height, args.width, device=device)
            ms, peak = time_model(model, x, args.residual_scale, device)
            rows.append({
                "model": name, "params": params, "precision": "fp32", "batch_size": batch,
                "resolution": f"{args.height}x{args.width}", "ms_per_frame": round(ms, 4), "peak_vram_mb": round(peak, 1),
            })
            log.append(f"{name} params={params} batch={batch} ms/frame={ms:.4f} peak_mb={peak:.1f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.output_root / "benchmark_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def batched_ms(name: str) -> float:
        return min(r["ms_per_frame"] for r in rows if r["model"] == name and r["batch_size"] > 1)

    summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "resolution": f"{args.height}x{args.width}",
        "reference": REFERENCE,
        "sanity": sanity,
        "batched_ms_per_frame": {name: batched_ms(name) for name in models},
        "runtime_budget_ms": 25.0,
        "budget_met": {name: bool(batched_ms(name) < 25.0) for name in models},
        "no_training": True,
        "no_teacher_inference": True,
    }
    (args.output_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "run.log").write_text("\n".join(log) + "\n")
    print(json.dumps(summary["batched_ms_per_frame"], indent=2))
    print(json.dumps({k: v.get("identity_at_init_max_abs_delta") for k, v in sanity.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
