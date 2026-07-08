#!/usr/bin/env python3
"""ARGOS v2 true causal streaming evaluator.

This is deliberately batch-size-1 semantic code: state is reset only at sequence
boundaries or explicit ablations, never at an arbitrary window boundary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path("/dtu/p1/leopam/ARGOS")
sys.path.insert(0, str(ROOT))

from scripts.temporal_refinement.causal_bida import (  # noqa: E402
    AlignedLocalOnlyFaithful,
    AlignedLocalOnlySafe,
    FaithfulCausalBiDA,
    SafeCausalBiDA,
)


TARGETS = ROOT / "results/03_temporal_refinement/evaluation/s2m2_gt_refiner_targets_full"
AUX_CACHE = ROOT / "results/03_temporal_refinement/nvds_lite_causal_pilot/aux_cache"


def warp_with_support(x: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward sampling: output(p)=x(p+flow(p)); returns in-bounds mask."""
    if flow.size(3) != 2:
        flow = flow.permute(0, 2, 3, 1)
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), -1).unsqueeze(0) + flow
    nx = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    ny = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    inb = ((nx.abs() <= 1) & (ny.abs() <= 1)).unsqueeze(1).float()
    y = F.grid_sample(x, torch.stack((nx, ny), -1), mode="bilinear", padding_mode="zeros", align_corners=True)
    return y, inb


def reliability_mask(current_valid: torch.Tensor, previous_valid: torch.Tensor | None, flow: torch.Tensor | None, occ: torch.Tensor | None) -> torch.Tensor:
    if previous_valid is None or flow is None:
        return current_valid.float()
    prev_warped, inb = warp_with_support(previous_valid.float(), flow)
    mask = current_valid.float() * (prev_warped > 0.5).float() * inb
    if occ is not None:
        mask = mask * (occ < 0.5).float()
    return mask * torch.isfinite(mask).float()


class RawIdentity(nn.Module):
    def init_state(self, *args, **kwargs):
        return None

    def detach_state(self, state):
        return None

    def step(self, current_rgb, current_raw_disparity, previous_rgb, previous_raw_disparity, flow_target_to_previous, reliability_mask, state):
        return current_raw_disparity, None, {"delta": torch.zeros_like(current_raw_disparity)}


class CurrentOnlySafe(SafeCausalBiDA):
    """Safe causal BiDA with no previous-frame evidence."""

    def step(self, current_rgb, current_raw_disparity, previous_rgb, previous_raw_disparity, flow_target_to_previous, reliability_mask, state):
        return super().step(current_rgb, current_raw_disparity, None, None, None, reliability_mask, state)


@dataclass
class StreamResult:
    refined: torch.Tensor
    diagnostics: list[dict[str, torch.Tensor]]
    frame_count: int
    temporal_pair_count: int


def stream_sequence(
    model: nn.Module,
    raw: torch.Tensor,
    valid: torch.Tensor,
    rgb: torch.Tensor | None = None,
    flow: torch.Tensor | None = None,
    occ: torch.Tensor | None = None,
    mode: str = "full",
    detach_state: bool = False,
) -> StreamResult:
    """Evaluate one full sequence chronologically.

    Shapes: raw/valid [T,1,H,W], rgb [T,3,H,W], flow [T-1,2,H,W], occ [T-1,1,H,W].
    """
    t, _, h, w = raw.shape
    state = model.init_state(1, h, w, raw.device, raw.dtype) if hasattr(model, "init_state") else None
    outputs, diags = [], []
    past = []
    for i in range(t):
        if mode == "state_reset":
            state = model.init_state(1, h, w, raw.device, raw.dtype) if hasattr(model, "init_state") else None
        if detach_state and state is not None and hasattr(model, "detach_state"):
            state = model.detach_state(state)

        current_raw = raw[i : i + 1]
        current_valid = valid[i : i + 1]
        current_rgb = None if rgb is None else rgb[i : i + 1]

        prev_raw = prev_rgb = flow_i = prev_valid = occ_i = None
        if i > 0 and mode != "current_only":
            if mode == "shuffled_history" and past:
                j = past[(i * 1103515245 + 12345) % len(past)]
                prev_raw = raw[j : j + 1]
                prev_rgb = None if rgb is None else rgb[j : j + 1]
                flow_i = torch.zeros_like(flow[0:1]) if flow is not None else None
                prev_valid = valid[j : j + 1]
                occ_i = None
            else:
                prev_raw = raw[i - 1 : i]
                prev_rgb = None if rgb is None else rgb[i - 1 : i]
                flow_i = None if flow is None else flow[i - 1 : i]
                prev_valid = valid[i - 1 : i]
                occ_i = None if occ is None else occ[i - 1 : i]
        rel = reliability_mask(current_valid, prev_valid, flow_i, occ_i)
        refined, state, diag = model.step(current_rgb, current_raw, prev_rgb, prev_raw, flow_i, rel, state)
        outputs.append(refined[0])
        diag = dict(diag)
        diag["reliability_mask"] = rel.detach()
        diags.append(diag)
        past.append(i)
    return StreamResult(torch.stack(outputs, 0), diags, t, max(t - 1, 0))


def aggregate_metrics(raw: torch.Tensor, refined: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    mask = valid > 0.5
    raw_err = (raw - gt).abs()
    ref_err = (refined - gt).abs()
    if not mask.any():
        return {"frames": float(raw.shape[0])}
    raw_good = mask & (raw_err < 3)
    return {
        "frames": float(raw.shape[0]),
        "raw_mae": float(raw_err[mask].mean()),
        "refined_mae": float(ref_err[mask].mean()),
        "delta_mae": float(raw_err[mask].mean() - ref_err[mask].mean()),
        "raw_bad3": float((raw_err[mask] >= 3).float().mean() * 100),
        "refined_bad3": float((ref_err[mask] >= 3).float().mean() * 100),
        "new_bad3": float(((ref_err >= 3) & raw_good).float().sum() / raw_good.float().sum().clamp_min(1) * 100),
        "modified_pixel_ratio": float(((refined - raw).abs()[mask] > 0.1).float().mean()),
    }


def load_scared_sequence(sequence_id: str | None = None, max_frames: int = 32, device: str = "cpu"):
    target_file = (TARGETS / "targets" / f"{sequence_id}.npz") if sequence_id else next((TARGETS / "targets").glob("*.npz"))
    aux_file = AUX_CACHE / target_file.name
    z, aux = np.load(target_file), np.load(aux_file)
    n = min(max_frames, z["raw_disp"].shape[0])
    raw = torch.from_numpy(z["raw_disp"][:n].astype("float32"))[:, None].to(device)
    gt = torch.from_numpy(z["gt_disp"][:n].astype("float32"))[:, None].to(device)
    valid = torch.from_numpy((z["valid_mask"][:n] > 0).astype("float32"))[:, None].to(device)
    rgb = torch.from_numpy(aux["rgb"][:n].astype("float32") / 255.0).permute(0, 3, 1, 2).to(device)
    flow = torch.from_numpy(aux["warp_flow"][: max(n - 1, 0)].astype("float32")).to(device)
    occ = torch.from_numpy(aux["occ"][: max(n - 1, 0)].astype("float32"))[:, None].to(device)
    return target_file.stem, raw, gt, valid, rgb, flow, occ


class ToyStateModel(nn.Module):
    def init_state(self, batch_size, height, width, device=None, dtype=None):
        return torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)

    def detach_state(self, state):
        return state.detach()

    def step(self, current_rgb, current_raw_disparity, previous_rgb, previous_raw_disparity, flow_target_to_previous, reliability_mask, state):
        if state is None:
            state = self.init_state(*current_raw_disparity.shape[:1], *current_raw_disparity.shape[-2:], current_raw_disparity.device, current_raw_disparity.dtype)
        if previous_raw_disparity is not None:
            state = state + 0.1 * previous_raw_disparity
        refined = current_raw_disparity + 0.01 * state
        return refined, state, {"hidden_state": state, "delta": refined - current_raw_disparity}


def run_self_tests() -> dict[str, object]:
    torch.manual_seed(3)
    raw = torch.arange(16, dtype=torch.float32).view(16, 1, 1, 1).expand(16, 1, 8, 8).clone()
    valid = torch.ones_like(raw)
    rgb = torch.zeros(16, 3, 8, 8)
    flow = torch.zeros(15, 2, 8, 8)
    occ = torch.zeros(15, 1, 8, 8)
    model = ToyStateModel()
    full = stream_sequence(model, raw, valid, rgb, flow, occ)
    first = stream_sequence(model, raw[:8], valid[:8], rgb[:8], flow[:7], occ[:7])
    state = model.init_state(1, 8, 8, raw.device, raw.dtype)
    prev = None
    outs = []
    for i in range(16):
        pr = None if i == 0 else raw[i - 1 : i]
        fl = None if i == 0 else flow[i - 1 : i]
        rel = reliability_mask(valid[i : i + 1], None if i == 0 else valid[i - 1 : i], fl, None if i == 0 else occ[i - 1 : i])
        y, state, _ = model.step(rgb[i : i + 1], raw[i : i + 1], None, pr, fl, rel, state)
        outs.append(y[0])
    chunk_preserved = torch.stack(outs, 0)
    reset8 = torch.cat([
        stream_sequence(model, raw[:8], valid[:8], rgb[:8], flow[:7], occ[:7]).refined,
        stream_sequence(model, raw[8:], valid[8:], rgb[8:], flow[8:], occ[8:]).refined,
    ])
    future = raw.clone()
    future[10:] += 999
    future_out = stream_sequence(model, future, valid, rgb, flow, occ).refined
    x = torch.zeros(1, 1, 8, 8)
    x[..., 2:6, 2:6] = 1
    cur = torch.zeros_like(x)
    cur[..., 2:6, 3:7] = 1
    fl = torch.zeros(1, 2, 8, 8)
    fl[:, 0] = -1
    warped, inb = warp_with_support(x, fl)
    pv = torch.ones_like(x)
    pv[..., :, :2] = 0
    rel = reliability_mask(torch.ones_like(x), pv, fl, torch.zeros_like(x))
    return {
        "sequence_continuity_no_reset_at_8": bool(not torch.allclose(full.refined[8:], reset8[8:])),
        "chunked_preserved_state_matches_full": bool(torch.allclose(full.refined, chunk_preserved, atol=0, rtol=0)),
        "incorrect_reset_negative_control_differs": bool(not torch.allclose(full.refined, reset8)),
        "future_perturbation_causal": bool(torch.allclose(full.refined[:10], future_out[:10], atol=0, rtol=0)),
        "state_reset_pair_counts": {"frames": full.frame_count, "temporal_pairs": full.temporal_pair_count},
        "flow_direction_improves_translation": bool((warped - cur).abs().mean() < (x - cur).abs().mean()),
        "warped_validity_nontrivial": bool(0 < float(rel.mean()) < 1),
        "finite_outputs": bool(torch.isfinite(full.refined).all()),
    }


def build_model(name: str) -> nn.Module:
    if name == "raw":
        return RawIdentity()
    if name == "current_only":
        return CurrentOnlySafe()
    if name == "aligned_local_faithful":
        return AlignedLocalOnlyFaithful(residual_bound=3.0)
    if name == "aligned_local_safe":
        return AlignedLocalOnlySafe()
    if name == "faithful_causal_bida":
        return FaithfulCausalBiDA(residual_bound=3.0)
    if name == "safe_causal_bida":
        return SafeCausalBiDA()
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="faithful_causal_bida", choices=["raw", "current_only", "aligned_local_faithful", "aligned_local_safe", "faithful_causal_bida", "safe_causal_bida"])
    ap.add_argument("--sequence-id", default=None)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--mode", default="full", choices=["full", "current_only", "shuffled_history", "state_reset"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        print(json.dumps(run_self_tests(), indent=2))
        return 0
    seq, raw, gt, valid, rgb, flow, occ = load_scared_sequence(args.sequence_id, args.max_frames, args.device)
    model = build_model(args.model).to(args.device).eval()
    with torch.no_grad():
        result = stream_sequence(model, raw, valid, rgb, flow, occ, mode=args.mode)
    print(json.dumps({"sequence": seq, "model": args.model, "mode": args.mode, **aggregate_metrics(raw, result.refined, gt, valid), "temporal_pairs": result.temporal_pair_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
