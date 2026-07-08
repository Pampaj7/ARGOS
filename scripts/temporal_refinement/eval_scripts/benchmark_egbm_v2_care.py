#!/usr/bin/env python3
"""Benchmark EGBM-v2-CARE vs EGBM-v1: params, runtime, VRAM, identity, warm start."""

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
from benchmark_modern_refiner_v4 import real_batch  # noqa: E402
from experimental_refiner_vx import egbm_refiner  # noqa: E402
from egbm_v2_care_refiner import egbm_v2_care, load_v1_warm_start  # noqa: E402


DEFAULT_V1_CHECKPOINT = Path("results/03_temporal_refinement/training/experimental_refiner_vx_training/checkpoints/best.pt")
DEFAULT_OUTPUT = Path("results/03_temporal_refinement/training/egbm_v2_care")
V1_REFERENCE = {"params": 4042068, "final_eval_ms_per_frame": 6.25, "gap_pct": 20.37}


@torch.no_grad()
def time_model(model, x, scale, device, warmup=10, iters=50):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    for _ in range(warmup):
        model(x, scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x, scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = 1000.0 * (time.perf_counter() - t0) / (iters * x.shape[0])
    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    return ms, peak


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--v1-checkpoint", type=Path, default=DEFAULT_V1_CHECKPOINT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--residual-scale", type=float, default=3.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    v1 = egbm_refiner(16, args.residual_scale).to(device).eval()
    v2 = egbm_v2_care(16, args.residual_scale).to(device).eval()
    loaded, total = (0, 0)
    if args.v1_checkpoint.exists():
        loaded, total = load_v1_warm_start(v2, str(args.v1_checkpoint))

    rows: list[dict[str, Any]] = []
    sanity: dict[str, Any] = {}
    x_real, raw = real_batch(args.targets_root, 4, device)
    for name, model in (("egbm_v1", v1), ("egbm_v2_care", v2)):
        params = sum(q.numel() for q in model.parameters())
        x = torch.randn(args.batch, 16, 256, 320, device=device)
        ms, peak = time_model(model, x, args.residual_scale, device)
        with torch.no_grad():
            out = model(x_real, args.residual_scale)
        bad_logit, p_bad, residual = out[:3]
        refined = raw + residual
        entry = {
            "params": params,
            "outputs_finite": bool(torch.isfinite(residual).all()),
            "identity_at_init_max_abs_delta_px": float((refined - raw).abs().max()),
        }
        if name == "egbm_v2_care":
            diag = out[3]
            entry["care_probs_shape"] = list(diag["care_probs"].shape)
            entry["surprise_shape"] = list(diag["surprise"].shape)
            entry["warm_start_tensors"] = f"{loaded}/{total}"
            # identity must hold even after warm start (expert heads still v1-trained!)
            # so this delta reflects warmed v1 behavior, not zero: report separately
        sanity[name] = entry
        rows.append({"model": name, "params": params, "batch": args.batch, "resolution": "256x320", "precision": "fp32", "ms_per_frame": round(ms, 4), "peak_vram_mb": round(peak, 1)})
        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.output_root / "benchmark_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ms_v2 = next(r["ms_per_frame"] for r in rows if r["model"] == "egbm_v2_care")
    summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "v1_reference": V1_REFERENCE,
        "rows": rows,
        "sanity": sanity,
        "estimated_system_total_ms_with_s2m2_62ms": round(62.0 + ms_v2, 2),
        "runtime_targets": {"preferred_15ms": bool(ms_v2 < 15.0), "acceptable_25ms": bool(ms_v2 < 25.0), "hard_35ms": bool(ms_v2 < 35.0)},
        "no_training": True,
        "no_teacher_inference": True,
    }
    (args.output_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_root / "run.log").open("a") as f:
        f.write(f"benchmark: v2_params={sanity['egbm_v2_care']['params']} ms={ms_v2} warm={loaded}/{total}\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
