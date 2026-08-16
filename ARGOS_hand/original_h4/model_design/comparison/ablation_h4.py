"""Evaluate a pre-registered ablation through the canonical inference path.

Same relationship to `canonical_h4_masked` as `seed_h4`: everything about the
evaluation is inherited --- the reset protocol, the evidence construction, the policy
thresholds, the support masking --- and only the two things the ablation actually
changed are swapped in, namely which head class is built and which cue builder is used.

Anything else varying would make the ablation non-comparable to the canonical run, which
is the whole point of the exercise.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from model_design.comparison.canonical_h4_masked import MaskedCanonicalH4

ROOT = Path(__file__).resolve().parents[2]
PREREGISTER = ROOT / "model_design/ablation_preregister.json"
RUNS = ROOT / "model_design/training_runs"
VARIANTS = {
    "A1_no_appearance": "A1_no_appearance_channels",
    "A2_no_learned_evidence": "A2_no_learned_stereo_evidence",
    "A3_single_resolution": "A3_single_resolution",
    "A4_relaxed_convexity": "A4_relaxed_convexity",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AblationH4(MaskedCanonicalH4):
    """Canonical masked inference, with one architectural change from the preregistration."""

    def __init__(self, *, variant: str, device: str = "cuda:0") -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}; expected one of {sorted(VARIANTS)}")
        declared = json.loads(PREREGISTER.read_text())["deviations_from_locked_recipe"]
        if VARIANTS[variant] not in declared:
            raise ValueError(f"{variant} is not pre-registered")
        self.variant = variant
        self.declared = declared[VARIANTS[variant]]
        self.checkpoint = RUNS / f"ablation_{variant}/checkpoints/best_validation.pt"
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"{variant} checkpoint missing: {self.checkpoint}")
        provenance = self.checkpoint.parents[1] / "ablation_provenance.json"
        if not provenance.is_file():
            raise FileNotFoundError(f"{variant} lacks its training provenance record")
        record = json.loads(provenance.read_text())
        if record["variant"] != variant:
            raise RuntimeError(f"provenance disagrees with requested variant {variant}")
        self.device = device
        self._model = self._extractor = self._build_cues = None
        self._policy = None
        self._provenance = record

    def describe(self) -> dict[str, Any]:
        return {
            "module": "ablation_h4",
            "variant_of": "canonical_h4_masked",
            "variant": self.variant,
            "declared_change": self.declared["change"],
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "training_provenance": self._provenance,
            "preregistration": str(PREREGISTER),
        }

    def _load(self):
        """Canonical construction, with the ablation's head class and cue builder."""
        if self._model is not None:
            return self._model, self._extractor, self._build_cues
        import sys
        import torch
        provenance = ROOT.parents[1] / "ARGOS_FREEZED/experiments/02_massive_training/scripts"
        if str(provenance) not in sys.path:
            sys.path.insert(0, str(provenance))
        from provenance.codd_style_fusion import (
            CODDStyleFusionHead, FrozenResNet18Layer1, build_codd_cues,
        )
        from model_design.comparison.ablation_variants import (
            RelaxedConvexityHead, SingleResolutionHead, build_cues_without_appearance,
        )

        head_class = {"A4_relaxed_convexity": RelaxedConvexityHead,
                      "A3_single_resolution": SingleResolutionHead}.get(
                          self.variant, CODDStyleFusionHead)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._model = head_class(state["cue_channels"]).to(self.device).eval().requires_grad_(False)
        self._model.load_state_dict(state["model"], strict=True)

        # A2 removes the learned evidence entirely, so it must not build the extractor:
        # leaving a zero-valued feature block would hand the head an architecture-side
        # marker that the trained variant never saw.
        needs_extractor = self.variant != "A2_no_learned_evidence"
        self._extractor = (FrozenResNet18Layer1().to(self.device).eval().requires_grad_(False)
                           if needs_extractor else None)
        self._build_cues = (build_cues_without_appearance
                            if self.variant == "A1_no_appearance" else build_codd_cues)
        expected = self.declared.get("cue_channels")
        if expected is not None and state["cue_channels"] != expected:
            raise RuntimeError(f"{self.variant}: checkpoint has {state['cue_channels']} cue "
                               f"channels, preregistration declares {expected}")
        return self._model, self._extractor, self._build_cues


def factory_a1(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A1_no_appearance", device=device)


def factory_a2(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A2_no_learned_evidence", device=device)


def factory_a3(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A3_single_resolution", device=device)


def factory_a4(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A4_relaxed_convexity", device=device)
