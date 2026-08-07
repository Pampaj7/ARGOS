#!/usr/bin/env python3
"""Fail-closed integrity verification for ARGOS v2 geometry-v1."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL = "40526a32ef6e9a62a3ea2b59e6751a60c441b8190f9b96522e3b12b35895d5cd"
EXPECTED_SEA = "1a21575ed6ca2c6945fb8e25c4169d241cf59ee5d12b8802c01c965206268cac"
EXPECTED_H4 = "99c5745c164fd4903b8aa8acf8f57efccacd9cdcd0d4ed4305cd10609324d725"
PROHIBITED = ("critic", "hard_negative", "spatial_safety", "codd_style", "ARGOS-V2", "model_design")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"FREEZE VERIFICATION FAILED: {message}")


def verify_hash_file() -> None:
    lines = (ROOT / "FILE_HASHES.sha256").read_text().splitlines()
    if not lines:
        fail("FILE_HASHES.sha256 is empty")
    for line in lines:
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.lstrip("* ")
        if not path.is_file() or sha256(path) != expected:
            fail(f"immutable hash mismatch: {path}")


def verify_imports() -> None:
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        lowered = "\n".join(imports).lower()
        for token in PROHIBITED:
            if token.lower() in lowered:
                fail(f"prohibited runtime import in {path}: {token}")


def main() -> None:
    manifest = json.loads((ROOT / "FREEZE_MANIFEST.json").read_text())
    if manifest.get("project_name") != "ARGOS v2" or manifest.get("frozen_version") != "geometry_v1":
        fail("manifest project/version mismatch")
    architecture = manifest["canonical_architecture"]
    if architecture.get("anchor_ages") != [1, 2, 4, 8]:
        fail("anchor ages changed")
    if architecture.get("fused_state_writeback") is not False or architecture.get("raw_anchor_writeback") is not True:
        fail("writeback contract changed")
    verify_hash_file()
    if sha256(ROOT / "checkpoints/raw_multi_anchor_best_validation.pt") != EXPECTED_MODEL:
        fail("copied multi-anchor checkpoint mismatch")
    external = json.loads((ROOT / "external_checkpoints/MANIFEST.json").read_text())["checkpoints"][0]
    if external["sha256"] != EXPECTED_SEA or sha256(Path(external["path"])) != EXPECTED_SEA:
        fail("SEA-RAFT checkpoint/provenance mismatch")
    h4 = json.loads((ROOT / "baselines/bounded_h4/MANIFEST.json").read_text())
    if h4["checkpoint_sha256"] != EXPECTED_H4 or sha256(Path(h4["checkpoint_path"])) != EXPECTED_H4:
        fail("H4 checkpoint/provenance mismatch")
    for relative, expected in manifest["configuration_hashes"].items():
        if sha256(ROOT / relative) != expected:
            fail(f"configuration hash mismatch: {relative}")
    for item in manifest["source_file_provenance"]:
        frozen = ROOT / item["frozen_relative_path"]
        if sha256(frozen) != item["frozen_sha256"]:
            fail(f"frozen provenance mismatch: {frozen}")
        if item.get("original_absolute_path"):
            source = Path(item["original_absolute_path"])
            if not source.is_file() or sha256(source) != item["original_sha256"]:
                fail(f"source provenance mismatch: {source}")
    commit = subprocess.check_output(
        ["git", "-C", manifest["source_repository_path"], "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != manifest["source_git_commit"]:
        fail(f"source commit changed: {commit}")
    verify_imports()
    sys.path.insert(0, str(ROOT / "src"))
    from argos_freezed.constants import ANCHOR_AGES
    if list(ANCHOR_AGES) != [1, 2, 4, 8]:
        fail("runtime anchor ages changed")
    print(f"PASS ARGOS v2 {manifest['frozen_version']} ({len((ROOT / 'FILE_HASHES.sha256').read_text().splitlines())} immutable hashes)")


if __name__ == "__main__":
    main()
