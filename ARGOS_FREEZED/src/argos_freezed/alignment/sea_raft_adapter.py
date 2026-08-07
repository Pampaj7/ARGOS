"""Frozen SEA-RAFT adapter extracted from the validated ARGOS v2 path."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from ..constants import EXTERNAL_ROOT, SEA_RAFT_CHECKPOINT
from .bida_pull_warp import _flow_bchw, resize_flow


class SEARAFTFlowAdapter:
    """Return direct target-to-source flow; never composes flow chains."""

    def __init__(self, *, checkpoint: str | Path = SEA_RAFT_CHECKPOINT, device=None, iterations: int = 4) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.iterations = iterations
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"SEA-RAFT checkpoint not found: {self.checkpoint}")
        external_root = str(EXTERNAL_ROOT)
        if external_root not in sys.path:
            sys.path.insert(0, external_root)
        raft_module = importlib.import_module("bidavideo.third_party.SEA-RAFT.core.raft")
        utils_module = importlib.import_module("bidavideo.third_party.SEA-RAFT.core.utils.utils")
        args = SimpleNamespace(use_var=True, var_min=0, var_max=10, pretrain="resnet18", initial_dim=64,
                               block_dims=[64, 128, 256], radius=4, dim=128, num_blocks=2, iters=iterations)
        self.model = raft_module.RAFT(args)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
        self._padder_cls = utils_module.InputPadder
        self.model.to(self.device).eval().requires_grad_(False)

    def infer(self, target_rgb: torch.Tensor, source_rgb: torch.Tensor, *, precomputed_flow: torch.Tensor | None = None,
              output_size: tuple[int, int] | None = None) -> torch.Tensor:
        if precomputed_flow is not None:
            flow = _flow_bchw(precomputed_flow).to(self.device, dtype=torch.float32)
            return resize_flow(flow, output_size) if output_size is not None else flow
        if target_rgb.shape != source_rgb.shape or target_rgb.ndim != 4:
            raise ValueError("target_rgb and source_rgb must share [B,3,H,W]")
        target = target_rgb.to(self.device, dtype=torch.float32)
        source = source_rgb.to(self.device, dtype=torch.float32)
        padder = self._padder_cls(target.shape)
        target_pad, source_pad = padder.pad(target, source)
        with torch.inference_mode():
            output = self.model(target_pad.contiguous(), source_pad.contiguous(), iters=self.iterations, test_mode=True)
            flow = padder.unpad(output["flow"][-1])
        return resize_flow(flow, output_size) if output_size is not None else flow

    def current_to_anchor(self, current_rgb: torch.Tensor, anchor_rgb: torch.Tensor) -> torch.Tensor:
        return self.infer(current_rgb, anchor_rgb)

    def anchor_to_current(self, anchor_rgb: torch.Tensor, current_rgb: torch.Tensor) -> torch.Tensor:
        return self.infer(anchor_rgb, current_rgb)
