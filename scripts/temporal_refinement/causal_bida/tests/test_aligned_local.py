#!/usr/bin/env python3
"""Minimal checks for ARGOS v2 aligned-local-only baselines."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.temporal_refinement.causal_bida import AlignedLocalOnlyFaithful, AlignedLocalOnlySafe  # noqa: E402
from scripts.temporal_refinement.causal_bida.official_blocks import flow_warp  # noqa: E402


def check_model(model: torch.nn.Module) -> None:
    torch.manual_seed(9)
    b, t, h, w = 2, 6, 24, 32
    raw = torch.rand(b, t, 1, h, w) * 8
    rgb = torch.rand(b, t, 3, h, w)
    valid = torch.ones(b, t, 1, h, w)
    flow = torch.zeros(b, t - 1, 2, h, w)
    out, _ = model.forward_sequence(raw, flow, rgb, valid)
    assert out.shape == raw.shape
    assert torch.isfinite(out).all()
    assert model.init_state(b, h, w, raw.device, raw.dtype) is None
    assert (out - raw).abs().max() < 1e-5
    if isinstance(model, AlignedLocalOnlySafe):
        _ref, _state, diag = model.step(rgb[:, 1], raw[:, 1], rgb[:, 0], raw[:, 0], flow[:, 0], valid[:, 1], None)
        assert 0 <= float(diag["gate"].min()) <= float(diag["gate"].max()) <= 1
        assert (diag["delta"].abs() <= model.residual_bound + 1e-5).all()
    model.zero_grad(set_to_none=True)
    loss = (model.forward_sequence(raw, flow, rgb, valid)[0] - raw).abs().mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def translation_support_check() -> None:
    x = torch.zeros(1, 1, 8, 8)
    x[..., 2:6, 2:6] = 1
    current = torch.zeros_like(x)
    current[..., 2:6, 3:7] = 1
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0] = -1  # target pixel samples one column left in previous frame.
    aligned = flow_warp(x, flow)
    assert (aligned - current).abs().mean() < (x - current).abs().mean()


def main() -> int:
    for model in (AlignedLocalOnlyFaithful(residual_bound=3.0), AlignedLocalOnlySafe()):
        check_model(model)
        print(type(model).__name__, "params", sum(p.numel() for p in model.parameters()))
    translation_support_check()
    print("aligned_local_tests=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
