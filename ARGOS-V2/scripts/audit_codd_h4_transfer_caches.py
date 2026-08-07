#!/usr/bin/env python3
"""Compact read-only cache contract audit for frozen ARGOS v2 H=4 transfer."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/codd_style_h4_transfer_audit/protocol_audit"
SCARED = ROOT / "cache_scaredc_backbones"
MULTI = ROOT / "cache_multidomain_backbones"
CHECKPOINT = ROOT / "results/codd_style_fusion_probe/bida_memory_phase1/full_phase1/seed_0/checkpoints/best_validation.pt"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def inspect(directory: Path, dataset: str, backbone: str) -> dict:
    required = ("disparity.npy", "valid_mask.npy", "frame_ids.npy", "metadata.json")
    row = {"dataset": dataset, "backbone": backbone, "path": str(directory),
           "complete": (directory / ".complete").exists(),
           "files_present": all((directory / name).exists() for name in required)}
    if not row["files_present"]:
        return row
    meta = json.loads((directory / "metadata.json").read_text())
    row["complete"] = bool(row["complete"] or meta.get("completion_status", False))
    disp = np.load(directory / "disparity.npy", mmap_mode="r")
    valid = np.load(directory / "valid_mask.npy", mmap_mode="r")
    ids = [str(item) for item in np.load(directory / "frame_ids.npy", allow_pickle=True).tolist()]
    manifest_path = directory / "frame_manifest.csv"
    manifests = list(csv.DictReader(manifest_path.open())) if manifest_path.exists() else []
    row.update({
        "frame_count": int(disp.shape[0]), "shape": "x".join(map(str, disp.shape)),
        "frame_ids_unique": len(ids) == len(set(ids)), "frame_manifest_count": len(manifests),
        "frame_ids_match_manifest": ([item.get("frame_id") for item in manifests] == ids) if manifests else "metadata_only",
        "finite": bool(np.isfinite(np.asarray(disp)).all()),
        "positive_where_valid": bool((np.asarray(disp)[np.asarray(valid).astype(bool)] > 0).all()),
        "valid_shape": tuple(disp.shape) == tuple(valid.shape),
        "cache_height": meta.get("cache_height"), "cache_width": meta.get("cache_width"),
        "units": meta.get("disparity_units"), "sign": meta.get("disparity_convention"),
        "source_width": meta.get("source_width"), "scale_formula": meta.get("disparity_scale_formula"),
        "rectified": meta.get("d4d_rectification", False), "checkpoint": meta.get("checkpoint_path", meta.get("checkpoint")),
    })
    if dataset == "D4D":
        groups = defaultdict(list)
        for index, item in enumerate(manifests): groups[item["sequence_id"]].append((index, int(item["temporal_order"])))
        row["sessions"] = len(groups)
        row["temporal_order_resets"] = int(sum(sum(group[i + 1][1] <= group[i][1] for i in range(len(group) - 1)) for group in groups.values()))
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for backbone_dir in sorted(SCARED.iterdir() if SCARED.exists() else []):
        if not backbone_dir.is_dir() or backbone_dir.name.startswith("_"): continue
        for sequence in sorted(backbone_dir.iterdir()):
            if sequence.is_dir(): rows.append(inspect(sequence, "SCARED-C", backbone_dir.name))
    for backbone_dir in sorted(MULTI.iterdir() if MULTI.exists() else []):
        if not backbone_dir.is_dir(): continue
        for dataset in sorted(backbone_dir.iterdir()):
            if dataset.is_dir(): rows.append(inspect(dataset, dataset.name, backbone_dir.name))
    fields = sorted({key for row in rows for key in row})
    with (OUT / "cache_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    compatible = [row for row in rows if row.get("complete") and row.get("files_present") and row.get("valid_shape") and row.get("finite") and row.get("positive_where_valid")]
    report = {
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": digest(CHECKPOINT),
        "cache_grid": [144, 180], "required_units": "pixels_at_cache_resolution",
        "compatible_cache_count": len(compatible), "cache_count": len(rows),
        "d4d_temporal_contract": "not directly ordered for H=4 replay" if any(row.get("temporal_order_resets", 0) for row in rows) else "ordered",
        "servct_temporal_contract": "static pairs only; H=4 temporal result not applicable",
        "rows": rows,
    }
    (OUT / "cache_compatibility.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
