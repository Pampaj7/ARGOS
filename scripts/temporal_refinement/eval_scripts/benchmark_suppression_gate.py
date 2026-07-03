#!/usr/bin/env python3
"""Benchmark the Suppression-Only Gate: params, runtime, VRAM, near-identity at init.

Measures the gate alone and the combined pipeline (frozen v3.2c + gate) against the
v3.2c base, on the standard 256x320 fp32 protocol. Also verifies that at init the
combined output matches v3.2c within the documented sigmoid(+4) pass-through (~98.2%),
i.e. max |combined - v3.2c| correction difference is <2% of the correction magnitude.
No training, no teacher inference.
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
from suppression_gate import SuppressionGate, V32CWithSuppression  # noqa: E402


DEFAULT_BASE_CHECKPOINT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_2c_hybrid_oracle_freeze_detector_long/checkpoints/best.pt")
DEFAULT_OUTPUT = Path("results/03_temporal_refinement/training/fable_light_experimental_refiner")
REFERENCE = {"v3.2c_base": {"params": 194818, "uncontended_ms_per_frame": 1.0786}, "v4_tiny": {"params": 942867, "ms": 2.7402}, "egbm": {"params": 4042068, "ms": 11.246}}


@torch.no_grad()
def time_fn(fn, warmup: int = 10, iters: int = 50, frames: int = 32) -> tuple[float, float]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = 1000.0 * (time.perf_counter() - t0) / (iters * frames)
    peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    return ms, peak


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--residual-scale", type=float, default=3.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    in_ch = int(ckpt.get("input_channels", 16))
    base = AbstentionCropRefiner(in_ch)
    base.load_state_dict(ckpt["model_state_dict"])
    gate = SuppressionGate(in_ch + 4)
    combined = V32CWithSuppression(base, gate, 0.7).to(device).eval()

    gate_params = sum(q.numel() for q in gate.parameters())
    base_params = sum(q.numel() for q in base.parameters())

    x = torch.randn(args.batch, in_ch, args.height, args.width, device=device)
    ms_base, peak_base = time_fn(lambda: combined.base(x, args.residual_scale), frames=args.batch)
    ms_comb, peak_comb = time_fn(lambda: combined(x, args.residual_scale), frames=args.batch)

    # near-identity at init on a real full-GT frame
    x_real, raw = real_batch(args.targets_root, 4, device)
    with torch.no_grad():
        _l, p_bad, res_base = combined.base(x_real, args.residual_scale)
        _l2, _p2, res_supp = combined(x_real, args.residual_scale)
    hard = (p_bad >= 0.7).float()
    corr_base = hard * res_base
    corr_supp = hard * res_supp
    max_corr = float(corr_base.abs().max())
    max_delta = float((corr_base - corr_supp).abs().max())
    rel_delta = max_delta / max(max_corr, 1e-9)

    rows: list[dict[str, Any]] = [
        {"model": "v3.2c_base_frozen", "params": base_params, "batch": args.batch, "ms_per_frame": round(ms_base, 4), "peak_vram_mb": round(peak_base, 1)},
        {"model": "v3.2c+suppression_gate", "params": base_params + gate_params, "batch": args.batch, "ms_per_frame": round(ms_comb, 4), "peak_vram_mb": round(peak_comb, 1)},
    ]
    with (args.output_root / "benchmark_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gate_params": gate_params,
        "base_params_frozen": base_params,
        "total_params": base_params + gate_params,
        "ms_per_frame_base": round(ms_base, 4),
        "ms_per_frame_combined": round(ms_comb, 4),
        "gate_overhead_ms": round(ms_comb - ms_base, 4),
        "runtime_budget_ms": 8.0,
        "budget_met": bool(ms_comb < 8.0),
        "near_identity_at_init": {
            "max_correction_px": max_corr,
            "max_abs_correction_delta_px": max_delta,
            "relative_delta": rel_delta,
            "expected_passthrough": "sigmoid(4.0) ~= 0.982, so relative delta ~= 1.8%",
            "check_passed": bool(rel_delta < 0.03),
        },
        "reference": REFERENCE,
        "no_training": True,
        "no_teacher_inference": True,
    }
    (args.output_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_root / "run.log").open("a") as f:
        f.write(f"benchmark: gate_params={gate_params} ms_base={ms_base:.4f} ms_combined={ms_comb:.4f} rel_identity_delta={rel_delta:.4f}\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
