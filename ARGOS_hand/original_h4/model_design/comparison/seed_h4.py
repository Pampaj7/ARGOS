"""Support-masked H=4 loaded from a pre-registered seed checkpoint.

`canonical_h4.py` pins its checkpoint path and verifies its hash, which is correct for the
canonical model and is left untouched. Seed variance requires evaluating the seed-1 and
seed-2 checkpoints produced under `multiseed_preregister.json` through the identical
inference path, so this module reuses `MaskedCanonicalH4` and overrides only which
checkpoint is loaded.

The architecture, the reset protocol, the evidence, the policy thresholds and the support
masking are all inherited unchanged. Only the weights differ, which is the whole point:
anything else varying would make the seeds non-comparable to seed 0.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from model_design.comparison.canonical_h4_masked import MaskedCanonicalH4

ROOT = Path(__file__).resolve().parents[2]
PREREGISTER = ROOT / "model_design/multiseed_preregister.json"
RUNS = ROOT / "model_design/training_runs"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SeedH4(MaskedCanonicalH4):
    """Identical inference path; weights come from a pre-registered seed run."""

    def __init__(self, *, seed: int, device: str = "cuda:0") -> None:
        allowed = json.loads(PREREGISTER.read_text())["deviation_from_locked_recipe"]["new_values"]
        if seed not in allowed:
            raise ValueError(f"seed {seed} is not pre-registered; allowed: {allowed}")
        self.seed = seed
        self.checkpoint = RUNS / f"canonical_h4_seed_{seed}/checkpoints/best_validation.pt"
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"seed {seed} checkpoint missing: {self.checkpoint}")
        provenance = self.checkpoint.parents[1] / "seed_provenance.json"
        if not provenance.is_file():
            raise FileNotFoundError(f"seed {seed} lacks its training provenance record")
        record = json.loads(provenance.read_text())
        if record["deviation"]["seed"]["used"] != seed:
            raise RuntimeError(f"seed provenance disagrees with requested seed {seed}")
        self.device = device
        self._model = self._extractor = self._build_cues = None
        self._policy = None
        self._provenance = record

    def describe(self) -> dict[str, Any]:
        return {
            "module": "seed_h4",
            "variant_of": "canonical_h4_masked",
            "seed": self.seed,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "training_provenance": self._provenance,
            "variant_change": "weights only; architecture, policy, reset protocol and support masking unchanged",
            "preregistration": str(PREREGISTER),
        }

    def _load(self):
        """Same construction as the canonical path, different weights."""
        if self._model is not None:
            return self._model, self._extractor, self._build_cues
        import sys
        import torch
        provenance = ROOT.parents[1] / "ARGOS_FREEZED/experiments/02_massive_training/scripts"
        if str(provenance) not in sys.path:
            sys.path.insert(0, str(provenance))
        from provenance.codd_style_fusion import CODDStyleFusionHead, FrozenResNet18Layer1, build_codd_cues
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._model = CODDStyleFusionHead(state["cue_channels"]).to(self.device).eval().requires_grad_(False)
        self._model.load_state_dict(state["model"], strict=True)
        self._extractor = FrozenResNet18Layer1().to(self.device).eval().requires_grad_(False)
        self._build_cues = build_codd_cues
        return self._model, self._extractor, self._build_cues


def factory_seed1(*, device: str = "cuda:0", **_: Any) -> SeedH4:
    return SeedH4(seed=1, device=device)


def factory_seed2(*, device: str = "cuda:0", **_: Any) -> SeedH4:
    return SeedH4(seed=2, device=device)


def factory(*, device: str = "cuda:0", **_: Any) -> SeedH4:
    return SeedH4(seed=int(os.environ.get("ARGOS_SEED", "1")), device=device)
