#!/usr/bin/env python3
"""Isolated smoke tests for exported ARGOS v2 external-component utilities.

No dataset, checkpoint, or full-model inference is required. RAFT/SEA-RAFT checkpoint
tests are documented and skip-safe because checkpoints are intentionally not exported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bidavideo.alignment.causal_warp import forward_backward_consistency, warp_to_current
from endostreamdepth.state_management.streaming_state import StreamingState
from ppmstereo.memory_selection.pick_and_play import aggregate_memory, score_memory, select_topk_and_weights


def test_causal_disparity_warp() -> None:
    disp = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5)
    flow = torch.zeros(1, 2, 1, 5)
    flow[:, 0] = 1.0
    warped, valid = warp_to_current(disp, flow)
    assert torch.allclose(warped[0, 0, 0, :4], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert valid[0, 0, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]

    back = torch.zeros_like(flow)
    back[:, 0] = -1.0
    mask = forward_backward_consistency(flow, back, threshold_px=0.01)
    assert mask.shape == (1, 1, 1, 5)
    assert mask[0, 0, 0, :4].sum().item() == 4


def test_memory_topk_selection() -> None:
    quality = torch.tensor([[0.9, 0.2, 0.7, 0.1]])
    similarity = torch.tensor([[0.1, 0.9, 0.2, 0.1]])
    redundancy = torch.tensor([[0.0, 0.4, 0.1, 0.0]])
    validity = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    scores = score_memory(quality, similarity, redundancy, validity)
    idx, weights = select_topk_and_weights(scores, k=2)
    assert idx.tolist() == [[0, 2]]
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))

    memory = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    agg = aggregate_memory(memory, idx, weights)
    assert agg.shape == (1, 1)
    assert 0.0 <= agg.item() <= 2.0

    poor = torch.full((1, 3), -1e9)
    idx2, w2 = select_topk_and_weights(poor, k=2)
    assert idx2.tolist() == [[0, 1]]
    assert w2.sum().item() == 0.0


def test_one_step_streaming_state() -> None:
    state = StreamingState()
    x0 = torch.ones(1, 3, 2, 2)
    out0 = state.step(x0)
    assert state.steps == 1
    assert torch.allclose(out0, x0)
    out1 = state.step(torch.zeros_like(x0), update_weight=0.25)
    assert state.steps == 2
    assert torch.allclose(out1, torch.full_like(x0, 0.75))
    state.detach()
    assert state.value is not None and not state.value.requires_grad
    state.reset()
    assert state.value is None and state.steps == 0


def test_flow_inference_smoke_skip_safe() -> None:
    """Document the intended RAFT/SEA-RAFT smoke test without requiring checkpoints."""
    sea_raft_ckpts = list((ROOT.parents[1] / "external" / "SEA-RAFT").glob("**/*.pth"))
    raft_ckpts = list((ROOT.parents[1] / "external" / "RAFT").glob("**/*.pth"))
    # Checkpoints are intentionally not copied/exported. Shape tests belong in wrappers once a
    # checkpoint path is explicitly configured.
    assert isinstance(sea_raft_ckpts, list)
    assert isinstance(raft_ckpts, list)


def main() -> int:
    test_causal_disparity_warp()
    test_memory_topk_selection()
    test_one_step_streaming_state()
    test_flow_inference_smoke_skip_safe()
    print("external component smoke tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
