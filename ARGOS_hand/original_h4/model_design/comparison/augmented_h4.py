"""The completed sequence-disjoint augmented H=4 checkpoint adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from model_design.comparison.canonical_h4 import CanonicalH4, ROOT


CHECKPOINT = ROOT / "model_design/checkpoints/h4_augmented/best_validation.pt"
PROVENANCE = CHECKPOINT.with_name("provenance.json")
CONFIGURATION = CHECKPOINT.with_name("configuration.json")
SPLIT_AUDIT = CHECKPOINT.with_name("split_audit.json")
CHECKPOINT_SHA256 = "83b3ee2fbd868ad1474318a4a812b68ac723166ddd9970a79fc84fcc30874409"
PROVENANCE_SHA256 = "56a311482e81028ffbbbac021d75843d0e1444015ad48d4763fed1b8846b5a66"
FROZEN_RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
PARAMETERS = 177_338
TRAIN_SEQUENCES = ("dataset_1_keyframe_2", "dataset_1_keyframe_3", "dataset_3_keyframe_1", "dataset_3_keyframe_2",
                   "dataset_3_keyframe_3", "dataset_3_keyframe_4", "dataset_6_keyframe_1", "dataset_6_keyframe_2",
                   "dataset_6_keyframe_3", "dataset_6_keyframe_4", "dataset_2_keyframe_2", "dataset_2_keyframe_3")
VALIDATION_SEQUENCES = ("dataset_2_keyframe_4",)
TEST_SEQUENCES = ("dataset_7_keyframe_1", "dataset_7_keyframe_2", "dataset_7_keyframe_3", "dataset_7_keyframe_4")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


class AugmentedH4(CanonicalH4):
    """Canonical causal H=4 semantics with the promoted augmented head."""

    def __init__(self, *, device: str = "cuda:0") -> None:
        self._provenance = self._verify_provenance()
        self.device = device
        self._model = self._extractor = self._build_cues = None

    @staticmethod
    def _verify_provenance() -> dict[str, Any]:
        if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
            raise RuntimeError(f"augmented checkpoint hash mismatch: {CHECKPOINT}")
        if _sha256(PROVENANCE) != PROVENANCE_SHA256:
            raise RuntimeError(f"augmented provenance hash mismatch: {PROVENANCE}")
        provenance, configuration, split = _json(PROVENANCE), _json(CONFIGURATION), _json(SPLIT_AUDIT)
        if provenance.get("profile") != "h4_augmented" or configuration.get("profile") != "h4_augmented":
            raise RuntimeError("augmented checkpoint requires the h4_augmented profile")
        if provenance.get("final_checkpoint_sha256") != CHECKPOINT_SHA256 or provenance.get("config") != configuration:
            raise RuntimeError("augmented checkpoint provenance does not bind its configuration")
        if provenance.get("split") != split:
            raise RuntimeError("augmented checkpoint provenance does not bind its split")
        expected = {"train_sequences": list(TRAIN_SEQUENCES), "validation_sequences": list(VALIDATION_SEQUENCES), "test_sequences": list(TEST_SEQUENCES),
                    "profile": "h4_augmented", "memory_state": "recurrent", "learned_stereo_evidence": True, "sequence_sets_pairwise_disjoint": True,
                    "frozen_resnet18_sha256": FROZEN_RESNET18_SHA256}
        if any(split.get(key) != value for key, value in expected.items()):
            raise RuntimeError("augmented checkpoint split contract mismatch")
        return provenance

    def describe(self) -> dict[str, Any]:
        return {"module": "augmented_h4", "checkpoint": str(CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA256,
                # `policy` keeps the existing definitive-run provenance verifier generic.
                "policy": str(PROVENANCE), "policy_sha256": PROVENANCE_SHA256,
                "provenance": str(PROVENANCE), "provenance_sha256": PROVENANCE_SHA256,
                "code": str(Path(__file__).resolve()), "code_sha256": _sha256(Path(__file__).resolve()),
                "reset_protocol": "fixed H=4; raw t-1 after re-anchor; preceding fused otherwise",
                "training_profile": "h4_augmented", "train_sequences": list(TRAIN_SEQUENCES),
                "validation_sequences": list(VALIDATION_SEQUENCES), "test_sequences": list(TEST_SEQUENCES),
                "evaluation_split_roles": {"SCARED-C/d2": "mixed_train_validation_exposed", "SCARED-C/d7": "heldout_test"}}

    def _load(self):
        if self._model is not None:
            return self._model, self._extractor, self._build_cues
        import torch
        from model_design.models.codd_style_fusion import CODDStyleFusionHead, FrozenResNet18Layer1, RESNET18_CHECKPOINT, build_codd_cues

        split = self._provenance["split"]
        if _sha256(RESNET18_CHECKPOINT) != FROZEN_RESNET18_SHA256:
            raise RuntimeError("augmented checkpoint frozen ResNet-18 provenance mismatch")
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        state_config = state.get("config")
        if not isinstance(state_config, dict):
            raise RuntimeError("augmented checkpoint has invalid configuration")
        # torch preserved the run output as Path; JSON provenance necessarily stores it as text.
        state_config = state_config | {"output": str(state_config["output"])}
        if state_config != self._provenance["config"] or state.get("split") != split or state.get("resnet_sha256") != split["frozen_resnet18_sha256"]:
            raise RuntimeError("augmented checkpoint state provenance mismatch")
        cue_channels = state.get("cue_channels")
        if not isinstance(cue_channels, int):
            raise RuntimeError("augmented checkpoint has invalid cue_channels")
        self._model = CODDStyleFusionHead(cue_channels).to(self.device)
        self._model.load_state_dict(state["model"], strict=True)
        if sum(parameter.numel() for parameter in self._model.parameters()) != PARAMETERS:
            raise RuntimeError("augmented checkpoint parameter count mismatch")
        self._model.eval().requires_grad_(False)
        self._extractor = FrozenResNet18Layer1().to(self.device).eval().requires_grad_(False)
        self._build_cues = build_codd_cues
        return self._model, self._extractor, self._build_cues


def factory(**kwargs: Any) -> AugmentedH4:
    return AugmentedH4(**kwargs)
