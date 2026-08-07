from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ZeroFlow:
    def __init__(self):
        self.calls = []

    def current_to_anchor(self, current, anchor):
        self.calls.append(("current_to_anchor", float(current.mean()), float(anchor.mean())))
        return torch.zeros(current.shape[0], 2, *current.shape[-2:], device=current.device)

    def anchor_to_current(self, anchor, current):
        self.calls.append(("anchor_to_current", float(anchor.mean()), float(current.mean())))
        return torch.zeros(current.shape[0], 2, *current.shape[-2:], device=current.device)


def raw(value=2.0, height=8, width=10):
    return torch.full((1, 1, height, width), float(value))


def rgb(value=10.0, height=8, width=10):
    return torch.full((1, 3, height, width), float(value))
