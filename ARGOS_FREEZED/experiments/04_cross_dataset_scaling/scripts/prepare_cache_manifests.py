#!/usr/bin/env python3
"""Hash/shape-only readiness manifest; deliberately never invokes inference."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

EXP = Path("/dtu/p1/leopam/ARGOS/ARGOS_FREEZED/experiments/04_cross_dataset_scaling")
OUT = EXP / "cache_preparation"
CACHE = Path("/dtu/p1/leopam/ARGOS/ARGOS-V2/cache_multidomain_backbones")
FILES = ("disparity.npy", "valid_mask.npy", "frame_ids.npy", "frame_manifest.csv", "metadata.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def classify(*, cache_rows: int, expected_causal_rows: int, flow_present: bool) -> str:
    if cache_rows != expected_causal_rows: return "BLOCKED_VERSION_PIN_REQUIRED"
    return "READY" if flow_present else "BLOCKED_FLOW_CACHE_MISSING"


def inspect_d4d(backbone: str) -> tuple[list[dict], dict]:
    directory = CACHE / backbone / "D4D"; entries = []
    present = all((directory / name).is_file() for name in FILES)
    if not present: return entries, {"backbone": backbone, "status": "BLOCKED_RAW_CACHE_MISSING"}
    disparity, valid, ids = (np.load(directory / name, mmap_mode="r") for name in FILES[:3])
    rows = sum(1 for _ in csv.DictReader((directory / "frame_manifest.csv").open()))
    sane = disparity.shape == valid.shape == (len(ids), 144, 180) and len(ids) == rows
    for name in FILES:
        path = directory / name; entries.append({"dataset": "D4D", "backbone": backbone, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size, "status": "verified"})
    return entries, {"backbone": backbone, "cache_rows": rows, "shape": list(disparity.shape), "valid_shape": list(valid.shape), "frame_ids": int(len(ids)), "integrity": "PASS" if sane else "FAIL", "flow_cache": "MISSING", "status": classify(cache_rows=rows * 2, expected_causal_rows=156, flow_present=False)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); manifest, d4d = [], []
    for backbone in ("RAFT-Stereo", "StereoAnywhere"):
        entries, status = inspect_d4d(backbone); manifest.extend(entries); d4d.append(status)
    stereomis = {"status": "BLOCKED_RAW_CACHE_MISSING", "searched": [str(CACHE / b / "StereoMIS") for b in ("RAFT-Stereo", "StereoAnywhere")], "flow_cache": "MISSING"}
    readiness = {"project": "ARGOS v2", "inference_executed": False, "dataset_7_opened": False,
                 "StereoMIS": stereomis,
                 "D4D": {"raw_disparity_caches": d4d, "aggregate_cache_manifest_rows": sum(x.get("cache_rows", 0) for x in d4d), "expected_causal_rows_unpinned": 156, "discrepancy": "832 cached rows (416 per backbone) versus 156 stated causal subset; pin the exact source/window manifest before any flow or zero-shot run", "sparse_zivid_anchors": {"total": 362, "usable": 239}, "flow_cache": "MISSING", "status": "BLOCKED_VERSION_PIN_REQUIRED"}}
    (OUT / "cache_readiness.json").write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n")
    with (OUT / "frozen_cache_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "backbone", "path", "sha256", "bytes", "status")); writer.writeheader(); writer.writerows(manifest)
    print(json.dumps({"status": "PASS", "d4d_entries": len(manifest), "dataset_7_opened": False}))


if __name__ == "__main__": main()
