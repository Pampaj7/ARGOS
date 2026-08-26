"""Evaluate a per-backbone specialist through the shipped head's own inference path.

A specialist is architecturally identical to A2: the shipped 38-channel head, no
learned stereo evidence, same reset protocol, same policy thresholds, same support
masking. Only the weights differ, because only the training data differed. So this
subclasses AblationH4 at variant A2 and redirects the checkpoint, which means the
head class, the cue builder and the extractor decision are inherited rather than
restated -- a second copy of that mapping is how a specialist would end up scored
on the 142-channel builder while its provenance claimed 38.

What it adds is the check that the redirected checkpoint really is a specialist:
its own provenance must name this backbone and must record the SEEN_BACKBONES
patch, so a directory holding some other run cannot be evaluated under a
specialist's name.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model_design.comparison.ablation_h4 import RUNS, AblationH4, _sha256

DECLARATION = Path(__file__).resolve().parents[2] / "model_design/specialist_control_declaration.json"


class SpecialistH4(AblationH4):
    """A2's inference path, reading the checkpoint trained on one backbone only."""

    def __init__(self, *, backbone: str, device: str = "cuda:0") -> None:
        declared = json.loads(DECLARATION.read_text())
        arm = f"specialist_{backbone}"
        if arm not in declared["arms"]:
            raise ValueError(f"{arm} is not a declared arm; allowed: {declared['arms']}")
        super().__init__(variant="A2_no_learned_evidence", device=device)

        self.backbone = backbone
        self.arm = arm
        self.checkpoint = RUNS / f"{arm}/checkpoints/best_validation.pt"
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"{arm} checkpoint missing: {self.checkpoint}")
        provenance = self.checkpoint.parents[1] / "specialist_provenance.json"
        if not provenance.is_file():
            raise FileNotFoundError(f"{arm} lacks its training provenance record")
        record = json.loads(provenance.read_text())
        if record.get("backbone") != backbone:
            raise RuntimeError(f"provenance says backbone {record.get('backbone')}, "
                               f"requested {backbone}")
        patched = record.get("deviation", {}).get("backbones", {})
        if patched.get("to") != [backbone]:
            raise RuntimeError(f"{arm} provenance does not record the single-backbone "
                               f"patch: {patched}")
        if record.get("deviation", {}).get("learned_stereo_evidence") is not False:
            raise RuntimeError(f"{arm} was not trained as a variant of the shipped head")
        self._provenance = record
        # super() resolved A2's checkpoint first; nothing may be cached from it.
        self._model = self._extractor = self._build_cues = None

    def describe(self) -> dict[str, Any]:
        value = super().describe()
        value.update({
            "module": "specialist_h4",
            "variant_of": "ablation_h4:A2_no_learned_evidence",
            "arm": self.arm,
            "backbone": self.backbone,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "training_provenance": self._provenance,
            "declaration": str(DECLARATION),
        })
        return value


def factory_raft(*, device: str = "cuda:0", **_: Any) -> SpecialistH4:
    return SpecialistH4(backbone="RAFT-Stereo", device=device)


def factory_stereoanywhere(*, device: str = "cuda:0", **_: Any) -> SpecialistH4:
    return SpecialistH4(backbone="StereoAnywhere", device=device)


def factory_s2m2(*, device: str = "cuda:0", **_: Any) -> SpecialistH4:
    return SpecialistH4(backbone="S2M2-S", device=device)


def factory_crestereo(*, device: str = "cuda:0", **_: Any) -> SpecialistH4:
    return SpecialistH4(backbone="CREStereo", device=device)


def factory_fastfoundation(*, device: str = "cuda:0", **_: Any) -> SpecialistH4:
    return SpecialistH4(backbone="Fast-FoundationStereo", device=device)
