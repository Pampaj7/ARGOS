"""Frozen adapter for the completed SCARED+C DRENDS H4 experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from model_design.comparison.canonical_h4 import CanonicalH4, ROOT

CHECKPOINT = ROOT / "model_design/checkpoints/h4_scared_drends_augmented/best_validation.pt"
PROVENANCE = CHECKPOINT.with_name("provenance.json")


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


class ScaredDrendsAugmentedH4(CanonicalH4):
    """Reuse the frozen canonical H4 causal lifecycle; only head weights differ."""
    def __init__(self, *, device: str = "cuda:0") -> None:
        if not CHECKPOINT.is_file() or not PROVENANCE.is_file(): raise FileNotFoundError("completed SCARED+DRENDS checkpoint/provenance required")
        self._provenance = json.loads(PROVENANCE.read_text())
        if self._provenance.get("profile") != "h4_scared_drends_augmented" or self._provenance.get("checkpoint_sha256") != _sha(CHECKPOINT): raise RuntimeError("SCARED+DRENDS H4 provenance mismatch")
        self.device = device; self._model = self._extractor = self._build_cues = None

    def describe(self) -> dict[str, Any]:
        return {"module": "h4_scared_drends_augmented", "checkpoint": str(CHECKPOINT), "checkpoint_sha256": _sha(CHECKPOINT), "policy": str(PROVENANCE), "policy_sha256": _sha(PROVENANCE), "code": str(Path(__file__)), "code_sha256": _sha(Path(__file__)), "reset_protocol": "fixed H=4; raw t-1 after re-anchor; preceding fused otherwise", "training_profile": "h4_scared_drends_augmented", "evaluation_split_roles": {"SCARED-C/d2": "mixed_train_validation_exposed", "SCARED-C/d7": "heldout_test", "DRENDS/Vid14_Pancreas_High": "historical_heldout_comparison"}}

    def _load(self):
        if self._model is not None: return self._model, self._extractor, self._build_cues
        import torch
        from model_design.models.codd_style_fusion import CODDStyleFusionHead, FrozenResNet18Layer1, build_codd_cues
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        state_config = state.get("config_identity") or {key: str(value) if isinstance(value, Path) else value for key, value in state.get("config", {}).items()}
        provenance_config = self._provenance.get("configuration_identity") or self._provenance.get("configuration")
        if state.get("split") != self._provenance.get("split") or state_config != provenance_config: raise RuntimeError("checkpoint state provenance mismatch")
        self._model = CODDStyleFusionHead(state["cue_channels"]).to(self.device).eval().requires_grad_(False); self._model.load_state_dict(state["model"], strict=True)
        self._extractor = FrozenResNet18Layer1().to(self.device).eval().requires_grad_(False); self._build_cues = build_codd_cues
        return self._model, self._extractor, self._build_cues


def factory(**kwargs: Any) -> ScaredDrendsAugmentedH4: return ScaredDrendsAugmentedH4(**kwargs)
