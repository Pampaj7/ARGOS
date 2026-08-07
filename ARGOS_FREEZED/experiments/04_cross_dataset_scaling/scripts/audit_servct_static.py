#!/usr/bin/env python3
"""CPU-only raw-cache static SERV-CT geometry audit; temporal modules are N/A."""
from __future__ import annotations
import csv, json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("/dtu/p1/leopam/ARGOS/ARGOS_FREEZED")
EXP = ROOT / "experiments/04_cross_dataset_scaling"
CACHE = Path("/dtu/p1/leopam/ARGOS/ARGOS-V2/cache_multidomain_backbones")


def resized_gt(sample: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    gt, valid = np.load(sample / "disp_gt.npy").astype(np.float32), np.load(sample / "valid_mask.npy").astype(bool)
    height, width = shape; coverage = cv2.resize(valid.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    numerator = cv2.resize(gt * valid, (width, height), interpolation=cv2.INTER_AREA)
    return numerator / np.maximum(coverage, 1e-6) * (width / gt.shape[1]), coverage > .5


def main() -> None:
    rows = []
    for backbone in ("RAFT-Stereo", "StereoAnywhere"):
        directory = CACHE / backbone / "SERV-CT"; required = [directory / x for x in ("disparity.npy", "valid_mask.npy", "frame_manifest.csv")]
        if not all(path.is_file() for path in required): continue
        disparity, valid = np.load(directory / "disparity.npy", mmap_mode="r"), np.load(directory / "valid_mask.npy", mmap_mode="r").astype(bool)
        manifest = list(csv.DictReader((directory / "frame_manifest.csv").open()))
        if len(manifest) != len(disparity): raise RuntimeError(f"cache length mismatch {backbone}")
        for index, item in enumerate(manifest):
            sample = Path(item["left_path"]).parent; gt, gt_valid = resized_gt(sample, disparity[index].shape)
            support = valid[index] & gt_valid & np.isfinite(disparity[index]) & np.isfinite(gt) & (disparity[index] > 0) & (gt > 0)
            error = np.abs(disparity[index][support] - gt[support])
            rows.append({"backbone": backbone, "frame_id": item["frame_id"], "valid_pixel_count": int(support.sum()), "raw_epe": float(error.mean()) if error.size else None, "temporal_refiners": "N/A: static pairs; no temporal adjacency", "dataset_7_opened": False})
    target = EXP / "servct_static_geometry"; target.mkdir(parents=True, exist_ok=True)
    with (target / "backbone_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["backbone", "frame_id", "valid_pixel_count", "raw_epe", "temporal_refiners", "dataset_7_opened"]); writer.writeheader(); writer.writerows(rows)
    (target / "README.md").write_text("# SERV-CT static geometry\n\nRaw cached stereo only. H4 and immutable temporal refiners are **not applicable** because frame ordering is not a temporal relation. CT-derived reference geometry is evaluated per static pair; no temporal claim is made.\n")
    (target / "camera_geometry.csv").write_text("dataset,temporal_status,reference\nSERV-CT,static_not_temporal,CT-derived registered disparity\n")
    (target / "aggregate_summary.json").write_text(json.dumps({"project": "ARGOS v2", "rows": len(rows), "raw_backbones": ["RAFT-Stereo", "StereoAnywhere"], "temporal_refiners": "N/A: static pairs; no temporal adjacency", "dataset_7_opened": False}, indent=2) + "\n")
    print(json.dumps({"project": "ARGOS v2", "rows": len(rows), "dataset_7_opened": False, "temporal_refiners": "N/A"}))


if __name__ == "__main__": main()
