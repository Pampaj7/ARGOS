"""Official RAFT inference wrapper for ARGOS v2 component probes.

Original repository: external/RAFT
Commit: 2888e15a51fa41140771d3afe71d03503b75
Original source paths:
- demo.py
- core/raft.py
- core/utils/utils.py

Checkpoint:
- default: external/frame_stereo_repos/RAFT/checkpoints/raft-things.pth

Tensor convention:
- image input [B,3,H,W], float 0..255 RGB
- output flow [B,2,H,W], pixels, image1 -> image2
- official InputPadder and RAFT upsampling are preserved.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
RAFT_ROOT = ROOT / "external/RAFT"
DEFAULT_CKPT = ROOT / "external/frame_stereo_repos/RAFT/checkpoints/raft-things.pth"


class RAFTArgs:
    small = False
    mixed_precision = False
    alternate_corr = False
    dropout = 0.0

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def _clear_raft_modules() -> None:
    for name in list(sys.modules):
        if name in {"raft", "corr", "extractor", "update"} or name.startswith("utils"):
            del sys.modules[name]


class OfficialRAFT:
    name = "RAFT"
    source = "external/RAFT/demo.py"
    checkpoint_requirement = str(DEFAULT_CKPT)

    def __init__(self, checkpoint: Path = DEFAULT_CKPT, device: str | None = None, iters: int = 20) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        self.checkpoint = checkpoint
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.iters = iters
        _clear_raft_modules()
        sys.path.insert(0, str(RAFT_ROOT / "core"))
        try:
            from raft import RAFT  # type: ignore
            from utils.utils import InputPadder  # type: ignore
        finally:
            try:
                sys.path.remove(str(RAFT_ROOT / "core"))
            except ValueError:
                pass
        self.InputPadder = InputPadder
        args = RAFTArgs()
        model = torch.nn.DataParallel(RAFT(args))
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        self.model = model.module.to(self.device).eval()

    @torch.no_grad()
    def infer(self, image1_rgb: np.ndarray, image2_rgb: np.ndarray) -> dict:
        image1 = torch.from_numpy(image1_rgb).permute(2, 0, 1).float()[None].to(self.device)
        image2 = torch.from_numpy(image2_rgb).permute(2, 0, 1).float()[None].to(self.device)
        padder = self.InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        _flow_low, flow_up = self.model(image1, image2, iters=self.iters, test_mode=True)
        flow_up = padder.unpad(flow_up)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        else:
            peak_mb = 0.0
        return {
            "flow": flow_up[0].detach().cpu().numpy().astype(np.float32),
            "runtime_ms": (time.perf_counter() - t0) * 1000.0,
            "peak_memory_mb": peak_mb,
        }


if __name__ == "__main__":
    print({"checkpoint_exists": DEFAULT_CKPT.exists(), "cuda": torch.cuda.is_available()})
