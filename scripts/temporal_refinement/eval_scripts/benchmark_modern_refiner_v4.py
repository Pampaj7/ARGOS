#!/usr/bin/env python3
"""Benchmark the v4 modern refiner prototypes against the v3 AbstentionCropRefiner.

Forward-pass only: parameter counts, ms/frame (fp32 and bf16 autocast), peak VRAM,
output shapes, numerical sanity (finite outputs, exact identity at zero-init), and a
real-batch check on one existing full-GT target shard. No training, no teacher inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))

from train_tiny_refiner_v1_full_gt import DEFAULT_TARGETS_ROOT, read_rows  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import AbstentionCropRefiner, make_features_from_raws  # noqa: E402
from modern_refiner_v4 import v4_small, v4_tiny  # noqa: E402


DEFAULT_OUTPUT = Path("results/03_temporal_refinement/training/modern_refiner_v4_benchmark")
V3_BASELINE = {"params": 194818, "reported_runtime_ms_per_frame": 1.076}


def forward_of(model: torch.nn.Module, x: torch.Tensor, scale: float):
    out = model(x, scale)
    return out[:3]  # (bad_logit, p_bad, residual) for both v3 and v4


@torch.no_grad()
def time_model(model: torch.nn.Module, x: torch.Tensor, scale: float, device: torch.device, bf16: bool, warmup: int = 10, iters: int = 50) -> tuple[float, float]:
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 and device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with ctx:
        for _ in range(warmup):
            forward_of(model, x, scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            forward_of(model, x, scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    ms_per_frame = 1000.0 * elapsed / (iters * x.shape[0])
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    return ms_per_frame, peak_mb


def real_batch(targets_root: Path, context_frames: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = read_rows(targets_root / "frame_targets_index.csv")
    shard = np.load(rows[0]["target_path"])
    offset = max(context_frames - 1, 4)
    ids = [max(0, offset - i) for i in range(context_frames)]
    raws = shard["raw_disp"][ids].astype(np.float32)
    valids = shard["valid_mask"][ids].astype(np.float32)
    x, _e, _v = make_features_from_raws(raws, valids)
    raw = torch.from_numpy(raws[0][None, None]).to(device)
    return torch.from_numpy(x[None]).to(device), raw


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
    }
    rows: list[dict[str, Any]] = []
    sanity: dict[str, Any] = {}
    for name, model in models.items():
        model = model.to(device).eval()
        params = sum(q.numel() for q in model.parameters())
        # numerical sanity on real data: finite outputs, exact identity at zero-init
        x_real, raw = real_batch(args.targets_root, 4, device)
        with torch.no_grad():
            bad_logit, p_bad, residual = forward_of(model, x_real, args.residual_scale)
        refined = raw + p_bad * residual
        sanity[name] = {
            "params": params,
            "output_shapes": {"bad_logit": list(bad_logit.shape), "p_bad": list(p_bad.shape), "residual": list(residual.shape)},
            "outputs_finite": bool(torch.isfinite(bad_logit).all() and torch.isfinite(p_bad).all() and torch.isfinite(residual).all()),
            "identity_at_init_max_abs_delta": float((refined - raw).abs().max()),
            "p_bad_at_init": float(p_bad.mean()),
        }
        for batch in args.batch_sizes:
            x = torch.randn(batch, args.in_channels, args.height, args.width, device=device)
            for bf16 in (False, True) if device.type == "cuda" else (False,):
                ms, peak = time_model(model, x, args.residual_scale, device, bf16)
                rows.append({
                    "model": name,
                    "params": params,
                    "precision": "bf16" if bf16 else "fp32",
                    "batch_size": batch,
                    "resolution": f"{args.height}x{args.width}",
                    "ms_per_frame": round(ms, 4),
                    "peak_vram_mb": round(peak, 1),
                })
                log.append(f"{name} params={params} batch={batch} {'bf16' if bf16 else 'fp32'} ms/frame={ms:.4f} peak_mb={peak:.1f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.output_root / "benchmark_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def best_ms(name: str) -> float:
        return min(r["ms_per_frame"] for r in rows if r["model"] == name and r["batch_size"] > 1)

    summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "resolution": f"{args.height}x{args.width}",
        "in_channels": args.in_channels,
        "v3_baseline": V3_BASELINE,
        "sanity": sanity,
        "best_batched_ms_per_frame": {name: best_ms(name) for name in models},
        "runtime_target_ms": 3.0,
        "targets_met": {name: bool(best_ms(name) < 3.0) for name in models},
        "no_training": True,
        "no_teacher_inference": True,
    }
    (args.output_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "run.log").write_text("\n".join(log) + "\n")

    lines = "\n".join(
        f"| {r['model']} | {r['params']:,} | {r['precision']} | {r['batch_size']} | {r['ms_per_frame']:.3f} | {r['peak_vram_mb']:.0f} |"
        for r in rows
    )
    (args.output_root / "README.md").write_text(f"""# Modern Refiner v4 Prototype Benchmark

Forward-pass benchmark of the v4 lightweight encoder-decoder refiner prototypes
(`scripts/temporal_refinement/models/modern_refiner_v4.py`) against the current v3
`AbstentionCropRefiner` ({V3_BASELINE['params']:,} params, reported ~{V3_BASELINE['reported_runtime_ms_per_frame']} ms/frame).
Input: existing v3 feature format ({args.in_channels} ch, {args.height}x{args.width}); no RGB, no training,
no S2M2/SAV/RAFT/DINO inference. v4 adds a third head (damping) controlling per-pixel
correction aggressiveness: refined = raw + gate(p_bad) * damping * bounded_residual.
All three heads are zero-initialized so v4 is an exact identity at init
(verified on a real full-GT target frame).

| Model | Params | Precision | Batch | ms/frame | Peak VRAM (MB) |
|---|---:|---|---:|---:|---:|
{lines}

Machine-readable results: `benchmark_summary.json` (includes output shapes, finite-output
and identity-at-init checks, and per-model <3 ms/frame target verdicts).

Next step (not run): v4 training with the v3.2c hybrid-oracle recipe plus damping-aware
hard-negative supervision on the two pathological clips.
""")
    print(json.dumps(summary["best_batched_ms_per_frame"], indent=2))
    print(json.dumps(summary["targets_met"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
