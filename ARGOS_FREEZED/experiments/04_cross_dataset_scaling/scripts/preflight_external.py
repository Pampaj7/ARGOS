#!/usr/bin/env python3
"""Fail-closed cache/provenance preflight; this script never runs inference."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

ROOT = Path("/dtu/p1/leopam/ARGOS/ARGOS_FREEZED")
EXP = ROOT / "experiments/04_cross_dataset_scaling"
CACHE = Path("/dtu/p1/leopam/ARGOS/ARGOS-V2/cache_multidomain_backbones")
READINESS = EXP / "cache_preparation/cache_readiness.json"


def audit(dataset: str) -> dict:
    if dataset not in {"StereoMIS", "D4D"}: raise ValueError("only StereoMIS or D4D")
    if "dataset_7" in dataset.lower(): raise RuntimeError("D7 ACCESS DENIED")
    caches = {name: CACHE / name / dataset for name in ("RAFT-Stereo", "StereoAnywhere")}
    required = ("disparity.npy", "valid_mask.npy", "frame_ids.npy", "frame_manifest.csv", "metadata.json")
    status = {name: {item: (path / item).is_file() for item in required} for name, path in caches.items()}
    complete = all(all(files.values()) for files in status.values())
    temporal = []
    if complete:
        for path in caches.values():
            with (path / "frame_manifest.csv").open() as handle:
                temporal.extend(csv.DictReader(handle))
    readiness_status = None
    if dataset == "D4D":
        readiness_status = (json.loads(READINESS.read_text())["D4D"]["status"] if READINESS.is_file()
                            else "BLOCKED_READINESS_MANIFEST_MISSING")
    result_status = "READY_FOR_SEPARATE_FROZEN_PROTOCOL" if complete else "BLOCKED_CACHE_INCOMPLETE"
    if readiness_status and readiness_status != "READY": result_status = readiness_status
    return {"project": "ARGOS v2", "dataset": dataset, "inference_started": False, "dataset_7_opened": False,
            "cache_root": str(CACHE), "required_cache_files": status,
            "temporal_manifest_rows": len(temporal),
            "cache_readiness_manifest": str(READINESS) if dataset == "D4D" else None,
            "cache_readiness_status": readiness_status, "status": result_status,
            "constraints": "no GT claim for StereoMIS; D4D sparse geometry requires verified Zivid registration and causal source window"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("dataset", choices=("StereoMIS", "D4D")); args = parser.parse_args()
    result = audit(args.dataset); target = EXP / ("stereomis_zero_shot" if args.dataset == "StereoMIS" else "d4d_zero_shot") / "PREFLIGHT.json"; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
