#!/usr/bin/env python3
"""Read-only check for the planning-only cross-dataset workspace."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REQUIRED = (
    "AGENTS.md", "DATASET_REGISTRY.yaml", "DATASET_AUDIT.md", "DATASET_CAPABILITY_MATRIX.csv",
    "DATASET_SPLITS.json", "DATA_USAGE_MANIFEST.json", "LEAKAGE_AUDIT.csv", "CAMERA_GEOMETRY.csv",
    "TEMPORAL_SUPPORT.csv", "ZERO_SHOT_PROTOCOL.json", "CROSS_TRAINING_PROTOCOL.json",
    "POOLED_TRAINING_PROTOCOL.json", "LODO_PROTOCOL.json", "scripts/README.md",
)

def main() -> None:
    assert all((HERE / name).is_file() for name in REQUIRED)
    usage = json.loads((HERE / "DATA_USAGE_MANIFEST.json").read_text())
    assert not any(usage["launches"].values())
    assert not (HERE / "TEST_UNLOCK").exists()
    assert not (HERE / "frozen_recipe_manifest").exists()
    print("PASS cross-dataset planning workspace: files present; launches and D7 remain blocked")

if __name__ == "__main__":
    main()
