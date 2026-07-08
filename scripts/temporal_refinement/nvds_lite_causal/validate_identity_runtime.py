#!/usr/bin/env python3
"""ARGOS v2 NVDS-lite identity-departure and runtime check."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/dtu/p1/leopam/ARGOS")
sys.path.insert(0, str(ROOT / "scripts/temporal_refinement/nvds_lite_causal"))
import train_nvds_lite as TR  # noqa: E402
from model import build_model  # noqa: E402

OUT = ROOT / "results/03_temporal_refinement/nvds_lite_causal_pilot/validation"


def pct(a, qs):
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} if a.size else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="A", choices=list(TR.CONFIGS))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    cfg = TR.CONFIGS[args.config]
    train = TR.load_split_shards("train")
    val = TR.load_split_shards("val")
    rng = np.random.default_rng(123)
    model = build_model(cfg["model"], cfg["use_rgb"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    step_times = []
    last_parts = {}
    for _ in range(args.steps):
        batch = [TR.sample_clip(train, args.clip_len, rng) for _ in range(args.batch)]
        raw, gt, valid, rgb, flow, occ = TR.collate(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            refined, _ = model(raw, valid, rgb, cfg["temporal_mode"], rng)
            loss, last_parts = TR.clip_losses(refined, raw, gt, valid, flow, occ, cfg)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

    model.eval()
    t0 = time.perf_counter()
    geo, tmp = TR.eval_sequences(model, val, device, cfg["temporal_mode"], rng, args.clip_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_s = time.perf_counter() - t0

    gate_vals, res_vals = [], []
    with torch.no_grad():
        for _ in range(8):
            batch = [TR.sample_clip(val, args.clip_len, rng)]
            raw, gt, valid, rgb, flow, occ = TR.collate(batch, device)
            refined, _ = model(raw, valid, rgb, cfg["temporal_mode"], rng)
            d = model.last_diag
            m = (valid > 0.5).detach().cpu().numpy().astype(bool)
            gate_vals.append(d["gate"].detach().cpu().numpy()[m])
            res_vals.append(np.abs(d["residual"].detach().cpu().numpy()[m]))
    gate = np.concatenate(gate_vals) if gate_vals else np.array([])
    res = np.concatenate(res_vals) if res_vals else np.array([])

    identity = {
        "config": args.config,
        "steps": args.steps,
        "clip_len": args.clip_len,
        "loss_terms_last": last_parts,
        "val_geometric": geo,
        "val_temporal": tmp,
        "modified_pixel_ratio": geo.get("modified_pixel_ratio"),
        "gate_mean": float(gate.mean()) if gate.size else float("nan"),
        "gate_std": float(gate.std()) if gate.size else float("nan"),
        "gate_percentiles": pct(gate, [1, 5, 50, 95, 99]),
        "residual_abs_mean": float(res.mean()) if res.size else float("nan"),
        "residual_abs_percentiles": pct(res, [50, 95, 99, 100]),
        "departed_from_identity": bool(geo.get("modified_pixel_ratio", 0.0) > 0.01 and (res.size and np.percentile(res, 95) > 0.01)),
        "not_saturated": bool((res.size == 0) or (np.mean(res > 2.95) < 0.01)),
    }
    runtime = {
        "config": args.config,
        "steps": args.steps,
        "batch": args.batch,
        "train_step_time_s_mean": float(np.mean(step_times)),
        "train_step_time_s_median": float(np.median(step_times)),
        "validation_time_s": float(eval_s),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0,
        "device": str(device),
    }
    (OUT / "identity_departure_validation.json").write_text(json.dumps(identity, indent=2, default=float) + "\n")
    (OUT / "runtime_validation.json").write_text(json.dumps(runtime, indent=2, default=float) + "\n")
    print(json.dumps({"identity": identity, "runtime": runtime}, indent=2, default=float))
    if not identity["departed_from_identity"]:
        raise SystemExit("identity departure check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
