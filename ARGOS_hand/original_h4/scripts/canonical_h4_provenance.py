"""No-Torch provenance checks shared by canonical bounded-H4 entry points."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "model_design/checkpoints/codd_style_h4_best_validation.pt"
POLICY = ROOT / "model_design/checkpoints/codd_style_h4_policy.json"
INFERENCE_MANIFEST = ROOT / "model_design/checkpoints/inference_manifest.json"
CHECKPOINT_SHA256 = "99c5745c164fd4903b8aa8acf8f57efccacd9cdcd0d4ed4305cd10609324d725"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inference_manifest() -> dict:
    manifest = json.loads(INFERENCE_MANIFEST.read_text())
    for name, record in manifest["artifacts"].items():
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"immutable artifact hash mismatch for {name}: {path}")
    return manifest


def verify_canonical_inputs(checkpoint: Path, policy: Path) -> dict:
    if checkpoint.resolve() != CHECKPOINT.resolve() or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError(f"canonical H4 checkpoint hash mismatch: {checkpoint}")
    if policy.resolve() != POLICY.resolve():
        raise RuntimeError(f"canonical H4 policy path mismatch: {policy}")
    value = json.loads(policy.read_text())
    expected = {
        "name": "fixed_h4", "max_age": 4,
        "accumulated_update_max": None, "disagreement_max": None,
        "warp_support_min": None, "fb_confidence_min": None,
        "temporal_activation_max": None, "update_magnitude_max": None,
    }
    if (value.get("selection_split") != "dataset_2_validation"
            or value.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or value.get("policy") != expected
            or value.get("hard_threshold") is not None):
        raise RuntimeError("policy is not the immutable dataset-2 selected fixed H=4 policy")
    verify_inference_manifest()
    return value
