#!/usr/bin/env python3
"""Tiny gradient-sanity training for the EGBM-Refiner (NOT a real training run).

Overfits ~300 steps on a handful of full-GT crops to verify: loss decreases, the model
leaves identity in the right direction (val refined MAE < raw on the overfit crops),
and every branch (experts, router, damping, boundary, dynamic threshold, memory)
receives gradient. Writes sanity_report.json into the benchmark output folder.
No teacher inference; full-GT targets only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))

from train_tiny_refiner_v1_full_gt import DEFAULT_TARGETS_ROOT, charbonnier, load_shards, masked_mean  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import make_features_from_raws  # noqa: E402
from train_tiny_refiner_v1_full_gt import read_rows  # noqa: E402
from experimental_refiner_vx import egbm_refiner  # noqa: E402


DEFAULT_OUTPUT = Path("results/03_temporal_refinement/training/experimental_refiner_vx_benchmark")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--crop", type=int, default=96)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rng = random.Random(0)
    rows = read_rows(args.targets_root / "frame_targets_index.csv")[:12]

    class _S:  # minimal Sample stand-in for load_shards
        def __init__(self, path):
            self.target_path = Path(path)

    shard = load_shards([_S(rows[0]["target_path"])])[Path(rows[0]["target_path"])]

    def sample_batch():
        xs, raws, gts, valids = [], [], [], []
        T, H, W = shard["raw_disp"].shape
        for _ in range(args.batch):
            t = rng.randrange(4, T)
            y = rng.randrange(0, H - args.crop)
            xx = rng.randrange(0, W - args.crop)
            ids = [max(0, t - i) for i in range(4)]
            rr = shard["raw_disp"][ids, y : y + args.crop, xx : xx + args.crop].astype(np.float32)
            vv = shard["valid_mask"][ids, y : y + args.crop, xx : xx + args.crop].astype(np.float32)
            feat, _e, _v = make_features_from_raws(rr, vv)
            xs.append(feat)
            raws.append(rr[0])
            gts.append(shard["gt_disp"][t, y : y + args.crop, xx : xx + args.crop].astype(np.float32))
            valids.append(vv[0])
        to = lambda a: torch.from_numpy(np.stack(a)).to(device)
        return to(xs), to(raws)[:, None], to(gts)[:, None], to(valids)[:, None]

    model = egbm_refiner().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    x, raw, gt, valid = sample_batch()
    with torch.no_grad():
        _l, _p, res0, _d = model(x, 3.0)
        raw_mae0 = float(masked_mean((raw - gt).abs(), valid))
    losses = []
    for step in range(args.steps):
        x, raw, gt, valid = sample_batch()
        _logit, p_bad, residual, diag = model(x, 3.0)
        refined = raw + residual
        raw_err = (raw - gt).abs()
        loss = (
            masked_mean(torch.clamp(charbonnier(refined - gt), max=10.0), valid)
            + 0.5 * masked_mean((residual).abs(), valid * (raw_err < 1.0).float())
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    # gradient coverage: every named branch must have received nonzero grads at least once
    grad_cover = {}
    for prefix in ("stem", "stage3", "memory_cell", "boundary_branch", "expert_heads", "router_head", "damping_head", "threshold_head"):
        has = any(p.grad is not None and float(p.grad.abs().sum()) >= 0 for n, p in model.named_parameters() if n.startswith(prefix))
        moved = any(float(p.abs().sum()) > 0 for n, p in model.named_parameters() if n.startswith(prefix) and "weight" in n)
        grad_cover[prefix] = {"in_graph": has, "weights_nonzero_after_training": moved}

    with torch.no_grad():
        x, raw, gt, valid = sample_batch()
        _l, _p, residual, diag = model(x, 3.0)
        refined = raw + residual
        raw_mae = float(masked_mean((raw - gt).abs(), valid))
        ref_mae = float(masked_mean((refined - gt).abs(), valid))

    report = {
        "steps": args.steps,
        "loss_first10_mean": float(np.mean(losses[:10])),
        "loss_last10_mean": float(np.mean(losses[-10:])),
        "loss_decreased": bool(np.mean(losses[-10:]) < np.mean(losses[:10])),
        "overfit_crops_raw_mae": raw_mae,
        "overfit_crops_refined_mae": ref_mae,
        "left_identity_in_right_direction": bool(ref_mae < raw_mae),
        "gradient_coverage": grad_cover,
        "diag_means_after_training": {
            "damping": float(diag["damping"].mean()),
            "gate": float(diag["gate"].mean()),
            "boundary_confidence": float(diag["boundary_confidence"].mean()),
            "identity_router_weight": float(diag["router_weights"][:, -1].mean()),
            "dynamic_threshold": float(diag["dynamic_threshold"].mean()),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "sanity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
