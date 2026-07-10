"""Official SEA-RAFT inference wrapper for ARGOS v2 component probes.

Original repository: external/SEA-RAFT
Commit: 9137517ba24e628442aec097d3afe71d03503b75
Original source paths:
- custom.py
- core/raft.py
- core/utils/utils.py
- config/parser.py

Checkpoint:
- default: external/bidavideo/third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-S.pth

Tensor convention:
- image input [B,3,H,W], float 0..255 RGB
- output flow [B,2,H,W], pixels, image1 -> image2
- preserves the official `custom.py::calc_flow` scale policy.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
SEA_ROOT = ROOT / "external/SEA-RAFT"
DEFAULT_CFG = SEA_ROOT / "config/train/Tartan-C-T-TSKH-spring540x960-S.json"
DEFAULT_CKPT = ROOT / "external/bidavideo/third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-S.pth"


def _clear_raft_modules() -> None:
    for name in list(sys.modules):
        if name in {"raft", "corr", "extractor", "update", "layer"} or name.startswith("utils") or name.startswith("config"):
            del sys.modules[name]


class OfficialSEARAFT:
    name = "SEA-RAFT"
    source = "external/SEA-RAFT/custom.py"
    checkpoint_requirement = str(DEFAULT_CKPT)

    def __init__(self, checkpoint: Path = DEFAULT_CKPT, cfg: Path = DEFAULT_CFG, device: str | None = None, scale_override: int = 0) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        self.checkpoint = checkpoint
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        _clear_raft_modules()
        sys.path.insert(0, str(SEA_ROOT))
        sys.path.insert(0, str(SEA_ROOT / "core"))
        try:
            from config.parser import json_to_args  # type: ignore
            from raft import RAFT  # type: ignore
            from utils.utils import load_ckpt  # type: ignore
        finally:
            for path in (str(SEA_ROOT / "core"), str(SEA_ROOT)):
                try:
                    sys.path.remove(path)
                except ValueError:
                    pass
        self.args = json_to_args(cfg)
        self.original_scale = getattr(self.args, "scale", 0)
        self.args.scale = scale_override
        self.model = RAFT(self.args)
        load_ckpt(self.model, checkpoint)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def infer(self, image1_rgb: np.ndarray, image2_rgb: np.ndarray) -> dict:
        image1 = torch.from_numpy(image1_rgb).permute(2, 0, 1).float()[None].to(self.device)
        image2 = torch.from_numpy(image2_rgb).permute(2, 0, 1).float()[None].to(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        scale = 2 ** getattr(self.args, "scale", 0)
        img1 = F.interpolate(image1, scale_factor=scale, mode="bilinear", align_corners=False)
        img2 = F.interpolate(image2, scale_factor=scale, mode="bilinear", align_corners=False)
        output = self.model(img1, img2, iters=self.args.iters, test_mode=True)
        flow = output["flow"][-1]
        info = output.get("info", [None])[-1]
        inv_scale = 0.5 ** getattr(self.args, "scale", 0)
        flow = F.interpolate(flow, scale_factor=inv_scale, mode="bilinear", align_corners=False) * inv_scale
        if info is not None:
            info = F.interpolate(info, scale_factor=inv_scale, mode="area")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        else:
            peak_mb = 0.0
        return {
            "flow": flow[0].detach().cpu().numpy().astype(np.float32),
            "info": None if info is None else info[0].detach().cpu().numpy().astype(np.float32),
            "runtime_ms": (time.perf_counter() - t0) * 1000.0,
            "peak_memory_mb": peak_mb,
        }


if __name__ == "__main__":
    print({"checkpoint_exists": DEFAULT_CKPT.exists(), "cuda": torch.cuda.is_available()})
