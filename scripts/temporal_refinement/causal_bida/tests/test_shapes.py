#!/usr/bin/env python3
"""Minimal synthetic checks for ARGOS v2 causal BiDA models."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.temporal_refinement.causal_bida import FaithfulCausalBiDA, SafeCausalBiDA  # noqa: E402


def assert_all_grads(model: torch.nn.Module, loss: torch.Tensor) -> None:
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not missing, missing
    assert not bad, bad


def run_model(model: torch.nn.Module) -> None:
    torch.manual_seed(7)
    b, t, h, w = 2, 5, 32, 40
    raw = torch.randn(b, t, 1, h, w).abs()
    rgb = torch.rand(b, t, 3, h, w)
    valid = torch.ones(b, t, 1, h, w)
    flow = torch.zeros(b, t - 1, 2, h, w)
    out, _ = model.forward_sequence(raw, flow, rgb, valid)
    assert out.shape == raw.shape
    assert torch.isfinite(out).all()
    if isinstance(model, SafeCausalBiDA):
        assert (out - raw).abs().max() <= model.residual_bound + 1e-5
    assert (out - raw).abs().max() < 1e-5, "identity init should be exact enough"

    out2, _ = model.forward_sequence(raw, flow, rgb, valid)
    raw_future = raw.clone()
    raw_future[:, 4] += 100
    out_future, _ = model.forward_sequence(raw_future, flow, rgb, valid)
    assert torch.allclose(out2[:, :4], out_future[:, :4], atol=0, rtol=0)

    state = model.init_state(b, h, w, raw.device, raw.dtype)
    flow1 = torch.zeros(b, 2, h, w)
    _y0, state0, _ = model.step(rgb[:, 0], raw[:, 0], None, None, None, valid[:, 0], state)
    _y1, state1, _ = model.step(rgb[:, 1], raw[:, 1], rgb[:, 0], raw[:, 0], flow1, valid[:, 1], state0)
    state_perturbed = state0.detach()
    state_perturbed.hidden = state_perturbed.hidden + 1.0
    _y1p, state1p, _ = model.step(rgb[:, 1], raw[:, 1], rgb[:, 0], raw[:, 0], flow1, valid[:, 1], state_perturbed)
    assert not torch.allclose(state1.hidden, state1p.hidden), "state perturbation should affect propagated state"

    model.zero_grad(set_to_none=True)
    out_train, _ = model.forward_sequence(raw, flow, rgb, valid)
    assert_all_grads(model, (out_train - raw).abs().mean())


def main() -> int:
    for model in (FaithfulCausalBiDA(residual_bound=3.0), SafeCausalBiDA()):
        run_model(model)
        params = sum(p.numel() for p in model.parameters())
        print(type(model).__name__, "params", params)
    print("causal_bida_synthetic_tests=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
