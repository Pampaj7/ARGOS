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
from typing import Any, Mapping

from model_design.comparison.canonical_h4 import CanonicalH4
from model_design.comparison.canonical_h4_masked import MaskedCanonicalH4

ROOT = Path(__file__).resolve().parents[2]
PREREGISTER = ROOT / "model_design/ablation_preregister.json"
RUNS = ROOT / "model_design/training_runs"
VARIANTS = {
    "A1_no_appearance": "A1_no_appearance_channels",
    "A2_no_learned_evidence": "A2_no_learned_stereo_evidence",
    "A3_single_resolution": "A3_single_resolution",
    "A4_relaxed_convexity": "A4_relaxed_convexity",
    "A5_no_fb_cue": "A5_no_fb_cue",
    "A6_geometry_only": "A6_geometry_only",
    "A7_half_width": "A7_half_width",
    "A3b_single_resolution_38ch": "A3b_single_resolution_38ch",
    "A4b_relaxed_convexity_38ch": "A4b_relaxed_convexity_38ch",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AblationH4(MaskedCanonicalH4):
    """Canonical masked inference, with one architectural change from the preregistration."""

    def __init__(self, *, variant: str, seed: int | None = None, device: str = "cuda:0") -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}; expected one of {sorted(VARIANTS)}")
        declared = json.loads(PREREGISTER.read_text())["deviations_from_locked_recipe"]
        if VARIANTS[variant] not in declared:
            raise ValueError(f"{variant} is not pre-registered")
        self.variant = variant
        self.seed = seed
        self.declared = declared[VARIANTS[variant]]
        # Seed runs live beside the seed-0 run rather than inside it, so the directory
        # carries the seed and the provenance is checked for it below.
        run = f"ablation_{variant}" + (f"_seed_{seed}" if seed is not None else "")
        self.checkpoint = RUNS / f"{run}/checkpoints/best_validation.pt"
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"{variant} checkpoint missing: {self.checkpoint}")
        provenance = self.checkpoint.parents[1] / "ablation_provenance.json"
        if not provenance.is_file():
            raise FileNotFoundError(f"{variant} lacks its training provenance record")
        record = json.loads(provenance.read_text())
        if record["variant"] != variant:
            raise RuntimeError(f"provenance disagrees with requested variant {variant}")
        used = record.get("deviation", {}).get("seed", {}).get("used")
        if seed is not None and used != seed:
            raise RuntimeError(f"provenance says seed {used}, requested {seed}")
        if seed is None and used is not None:
            raise RuntimeError(f"{variant} without a seed resolved to a seed-{used} run")
        self.device = device
        self._model = self._extractor = self._build_cues = None
        self._policy = None
        self._provenance = record

    def describe(self) -> dict[str, Any]:
        return {
            "module": "ablation_h4",
            "variant_of": "canonical_h4_masked",
            "variant": self.variant,
            "seed": self.seed,
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
            HalfWidthHead, RelaxedConvexityHead, SingleResolutionHead,
            build_cues_without_appearance, build_cues_without_learned_evidence,
            build_cues_without_fb_confidence, build_cues_geometry_only,
        )

        head_class = {"A4_relaxed_convexity": RelaxedConvexityHead,
                      "A3_single_resolution": SingleResolutionHead,
                      "A7_half_width": HalfWidthHead,
                      "A3b_single_resolution_38ch": SingleResolutionHead,
                      "A4b_relaxed_convexity_38ch": RelaxedConvexityHead}.get(
                          self.variant, CODDStyleFusionHead)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._model = head_class(state["cue_channels"]).to(self.device).eval().requires_grad_(False)
        self._model.load_state_dict(state["model"], strict=True)

        # Which base a variant deviates from decides both of the next two choices, so
        # they are driven by one list. Only the three variants of the 142-channel base
        # build the frozen extractor; every variant of the shipped head removes the
        # learned evidence entirely and must not build it, because a zero-valued feature
        # block would hand the head an architecture-side marker it never saw in training.
        base_142 = ("A1_no_appearance", "A3_single_resolution", "A4_relaxed_convexity")
        needs_extractor = self.variant in base_142
        self._extractor = (FrozenResNet18Layer1().to(self.device).eval().requires_grad_(False)
                           if needs_extractor else None)
        # Every variant of the shipped head TRAINED on 38 channels, so leaving any of them
        # on the canonical 142-channel builder would score the head on inputs it never saw.
        self._build_cues = {"A1_no_appearance": build_cues_without_appearance,
                            "A2_no_learned_evidence": build_cues_without_learned_evidence,
                            "A5_no_fb_cue": build_cues_without_fb_confidence,
                            "A6_geometry_only": build_cues_geometry_only,
                            "A7_half_width": build_cues_without_learned_evidence,
                            "A3b_single_resolution_38ch": build_cues_without_learned_evidence,
                            "A4b_relaxed_convexity_38ch": build_cues_without_learned_evidence,
                            }.get(self.variant, build_codd_cues)
        expected = self.declared.get("cue_channels")
        if expected is not None and state["cue_channels"] != expected:
            raise RuntimeError(f"{self.variant}: checkpoint has {state['cue_channels']} cue "
                               f"channels, preregistration declares {expected}")
        return self._model, self._extractor, self._build_cues


def factory_a1(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A1_no_appearance", device=device)


def factory_a2(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A2_no_learned_evidence", device=device)


def factory_a2_seed1(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A2_no_learned_evidence", seed=1, device=device)


def factory_a2_seed2(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A2_no_learned_evidence", seed=2, device=device)


def factory_a3(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A3_single_resolution", device=device)


def factory_a3_seed1(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A3_single_resolution", seed=1, device=device)


def factory_a3_seed2(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A3_single_resolution", seed=2, device=device)


def factory_a4_seed1(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A4_relaxed_convexity", seed=1, device=device)


def factory_a4_seed2(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A4_relaxed_convexity", seed=2, device=device)


def factory_a4(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A4_relaxed_convexity", device=device)


def factory_a5(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A5_no_fb_cue", device=device)


def factory_a6(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A6_geometry_only", device=device)


def factory_a7(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A7_half_width", device=device)


def factory_a3b(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A3b_single_resolution_38ch", device=device)


def factory_a4b(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A4b_relaxed_convexity_38ch", device=device)


class UnconfinedAblationH4(AblationH4):
    """The shipped head with the support contract removed from its OUTPUT only.

    `AblationH4` inherits `MaskedCanonicalH4.step`, which applies
    `fused = where(support, fused, raw)`. This skips that one line by calling the
    unmasked parent directly, so the support is still computed and reported and only the
    confinement is gone. That is the same single-line ablation the paper reports on the
    142-channel configuration, now runnable on the head the paper proposes and on exactly
    the same cells, which is what makes the two arms paired.
    """

    def describe(self) -> dict[str, Any]:
        return super().describe() | {"variant_change": "support computed but NOT applied to the output",
                                     "masked_to_support": False}

    def step(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        # Deliberately bypasses MaskedCanonicalH4.step in the MRO.
        return CanonicalH4.step(self, frame)


def factory_a2_unconfined(*, device: str = "cuda:0", **_: Any) -> UnconfinedAblationH4:
    return UnconfinedAblationH4(variant="A2_no_learned_evidence", device=device)


class A5WithRealCue(AblationH4):
    """A5's checkpoint fed the real forward-backward confidence it trained without.

    The shipped head trained WITH the cue ignores it at inference (constant substitution
    moves the D2 closure by less than the seed spread). This is the converse arm: if the
    head trained WITHOUT the cue is equally indifferent to receiving the real one, the
    indifference is a property of the architecture at inference rather than an accident
    of one checkpoint, and the train-with/deploy-without claim stops being anecdotal.
    """

    def _load(self):
        model, extractor, _ = super()._load()
        from model_design.comparison.ablation_variants import build_cues_without_learned_evidence
        self._build_cues = build_cues_without_learned_evidence
        return model, extractor, self._build_cues

    def describe(self):
        return super().describe() | {"cue_override": "real C^FB fed to the A5 head at inference"}


def factory_a5_realfb(*, device: str = "cuda:0", **_: Any) -> A5WithRealCue:
    return A5WithRealCue(variant="A5_no_fb_cue", device=device)


def factory_a3b_seed1(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A3b_single_resolution_38ch", seed=1, device=device)


def factory_a3b_seed2(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A3b_single_resolution_38ch", seed=2, device=device)


def factory_a6_seed1(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A6_geometry_only", seed=1, device=device)


def factory_a6_seed2(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A6_geometry_only", seed=2, device=device)


def factory_a1_seed1(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A1_no_appearance", seed=1, device=device)


def factory_a1_seed2(*, device: str = "cuda:0", **_: Any) -> AblationH4:
    return AblationH4(variant="A1_no_appearance", seed=2, device=device)
