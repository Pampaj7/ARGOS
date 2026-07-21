#!/usr/bin/env python3
"""Build a separate frozen-backbone cache for genuinely supervised OOD frames.

This deliberately supports only D4D curated-anchor contexts and prepared
SERV-CT pairs.  It is a prerequisite for a backbone-balanced multi-domain
experiment; it does not read GT, train anything, or alter ARGOS v2 modules.

Run one backbone per process because the validated external wrappers have
incompatible import namespaces.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from argos_v2.backbones import build_predictor, inference_resolution
from argos_v2.paths import ARGOS_ROOT, CACHE_HEIGHT, CACHE_WIDTH


DEFAULT_OUT = ARGOS_ROOT / "ARGOS-V2/cache_multidomain_backbones"
D4D_CONTEXT = ARGOS_ROOT / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot"
D4D_GT_MANIFEST = (ARGOS_ROOT / "dataset/D4D/processed/keyframe_stereo_gt_curated/"
                   "manifests/valid_and_warning_manifest.csv")
SERV_ROOT = ARGOS_ROOT / "results/03_temporal_refinement/ood/prepared/servct"

CHECKPOINTS = {
    "RAFT-Stereo": ARGOS_ROOT / "external/frame_stereo_repos/RAFT-Stereo/models/raftstereo-middlebury.pth",
    "StereoAnywhere": ARGOS_ROOT / "external/frame_stereo_repos/stereoanywhere/weights/stereoanywhere_sceneflow.pth",
}


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    sequence_id: str
    left_path: str
    right_path: str
    domain: str
    specimen: str
    session: str
    temporal_order: int
    rectified: bool


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partial_hash(path: Path, n_bytes: int = 8_000_000) -> str:
    if not path.exists():
        return "unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(n_bytes))
    return digest.hexdigest()[:16]


def git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ARGOS_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _d4d_imports():
    source = ARGOS_ROOT / "scripts/temporal_refinement/ood/d4d"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from d4d_keyframe_gt import load_cam, rectify_maps, session_root
    return load_cam, rectify_maps, session_root


def d4d_records(limit_anchors: int = 0, specimens: set[str] | None = None) -> list[FrameRecord]:
    """Return unique causal context frames in current-to-past source order."""
    contexts = {row["anchor_id"]: row for row in csv.DictReader((D4D_CONTEXT / "context_manifest.csv").open())}
    anchors = list(csv.DictReader(D4D_GT_MANIFEST.open()))
    anchors = [row for row in anchors if row["anchor_id"] in contexts and row.get("right_rectified_path")]
    if specimens:
        anchors = [row for row in anchors if row["specimen_id"] in specimens]
    anchors.sort(key=lambda row: row["anchor_id"])
    if limit_anchors:
        anchors = anchors[:limit_anchors]

    load_cam, rectify_maps, session_root = _d4d_imports()
    map_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    output: list[FrameRecord] = []
    seen: set[str] = set()
    for anchor in anchors:
        specimen, session = anchor["specimen_id"], anchor["session_id"]
        key = (specimen, session)
        if key not in map_cache:
            camera_dir = session_root(specimen) / session / "camera_info"
            left_cam, right_cam = load_cam(camera_dir / "left.yaml"), load_cam(camera_dir / "right.yaml")
            left_map, right_map = rectify_maps(left_cam), rectify_maps(right_cam)
            map_cache[key] = (left_map[0], left_map[1], right_map[0], right_map[1])
        stems = contexts[anchor["anchor_id"]]["context_stems"].split(";")
        if len(stems) != 4:
            raise RuntimeError(f"D4D anchor {anchor['anchor_id']} has non-causal context: {stems}")
        root = session_root(specimen) / session
        for temporal_order, stem in enumerate(stems):  # t, t-1, t-2, t-3
            frame_id = f"{specimen}__{session}__{stem}"
            if frame_id in seen:
                continue
            left_path, right_path = root / "left_images" / f"{stem}.png", root / "right_images" / f"{stem}.png"
            if not left_path.exists() or not right_path.exists():
                raise FileNotFoundError(f"missing D4D stereo pair for {frame_id}")
            if left_path.resolve() == right_path.resolve():
                raise RuntimeError(f"D4D source views alias each other: {frame_id}")
            seen.add(frame_id)
            output.append(FrameRecord(frame_id, f"{specimen}__{session}", str(left_path), str(right_path),
                                      "D4D", specimen, session, temporal_order, True))
    if not output:
        raise RuntimeError("no D4D frames selected")
    return output


def serv_records(limit_sequences: int = 0) -> list[FrameRecord]:
    rows = list(csv.DictReader((SERV_ROOT / "sequence_manifest.csv").open()))
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["sequence_id"], []).append(row)
    sequences = sorted(groups)
    if limit_sequences:
        sequences = sequences[:limit_sequences]
    output: list[FrameRecord] = []
    for sequence in sequences:
        sequence_rows = sorted(groups[sequence], key=lambda row: int(row["order_index"]))
        for order, row in enumerate(sequence_rows):
            left_path, right_path = Path(row["left_path"]), Path(row["right_path"])
            if not left_path.exists() or not right_path.exists():
                raise FileNotFoundError(f"missing SERV-CT stereo pair {sequence}/{row['frame_id']}")
            if left_path.resolve() == right_path.resolve():
                raise RuntimeError(f"SERV-CT source views alias each other: {sequence}/{row['frame_id']}")
            output.append(FrameRecord(f"{sequence}__{row['frame_id']}", sequence, str(left_path), str(right_path),
                                      "SERV-CT", sequence, sequence, order, False))
    if not output:
        raise RuntimeError("no SERV-CT frames selected")
    return output


def read_pair(record: FrameRecord, d4d_map_cache: dict[tuple[str, str], tuple[np.ndarray, ...]]) -> tuple[np.ndarray, np.ndarray]:
    left = cv2.imread(record.left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(record.right_path, cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise RuntimeError(f"cannot read stereo pair for {record.frame_id}")
    if record.rectified:
        key = (record.specimen, record.session)
        if key not in d4d_map_cache:
            load_cam, rectify_maps, session_root = _d4d_imports()
            camera_dir = session_root(record.specimen) / record.session / "camera_info"
            left_cam, right_cam = load_cam(camera_dir / "left.yaml"), load_cam(camera_dir / "right.yaml")
            left_map, right_map = rectify_maps(left_cam), rectify_maps(right_cam)
            d4d_map_cache[key] = (left_map[0], left_map[1], right_map[0], right_map[1])
        lm_x, lm_y, rm_x, rm_y = d4d_map_cache[key]
        left = cv2.remap(left, lm_x, lm_y, cv2.INTER_LINEAR)
        right = cv2.remap(right, rm_x, rm_y, cv2.INTER_LINEAR)
    if left.shape != right.shape:
        raise RuntimeError(f"left/right shape mismatch for {record.frame_id}: {left.shape} vs {right.shape}")
    return cv2.cvtColor(left, cv2.COLOR_BGR2RGB), cv2.cvtColor(right, cv2.COLOR_BGR2RGB)


def resize_to_cache(disparity: np.ndarray, native_width: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(disparity) & (disparity > 0)
    clean = np.where(valid, disparity, 0).astype(np.float32)
    resized = cv2.resize(clean, (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA)
    output = resized * (CACHE_WIDTH / float(native_width))
    valid_out = cv2.resize(valid.astype(np.uint8), (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return output.astype(np.float16), valid_out.astype(np.uint8)


def validate_cache(directory: Path, records: list[FrameRecord]) -> dict[str, object]:
    required = ("disparity.npy", "valid_mask.npy", "frame_ids.npy", "frame_manifest.csv", "metadata.json")
    checks: dict[str, object] = {"files_exist": all((directory / item).exists() for item in required)}
    if not checks["files_exist"]:
        checks["passed"] = False
        return checks
    disparity = np.load(directory / "disparity.npy", mmap_mode="r")
    valid = np.load(directory / "valid_mask.npy", mmap_mode="r")
    frame_ids = np.load(directory / "frame_ids.npy").tolist()
    expected = [record.frame_id for record in records]
    checks.update({
        "shape": tuple(disparity.shape),
        "shape_ok": tuple(disparity.shape) == (len(records), CACHE_HEIGHT, CACHE_WIDTH),
        "valid_shape_ok": tuple(valid.shape) == tuple(disparity.shape),
        "dtype_ok": disparity.dtype == np.float16 and valid.dtype == np.uint8,
        "frame_ids_exact": frame_ids == expected,
        "finite": bool(np.isfinite(np.asarray(disparity)).all()),
        "positive_where_valid": bool((np.asarray(disparity)[np.asarray(valid).astype(bool)] > 0).all()),
    })
    checks["passed"] = bool(all(checks[key] for key in ("shape_ok", "valid_shape_ok", "dtype_ok", "frame_ids_exact", "finite", "positive_where_valid")))
    return checks


def write_manifest(path: Path, records: list[FrameRecord]) -> None:
    fields = list(FrameRecord.__dataclass_fields__)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: getattr(record, field) for field in fields} for record in records)


def build(records: list[FrameRecord], *, backbone: str, output: Path, device: torch.device, force: bool) -> None:
    final = output / backbone / records[0].domain
    if final.exists() and (final / ".complete").exists() and not force:
        check = validate_cache(final, records)
        if check.get("passed"):
            print(f"SKIP validated cache: {final}")
            return
        raise RuntimeError(f"completed cache failed validation; use --force after investigation: {final}")
    if final.exists() and not force:
        raise FileExistsError(f"refusing to modify existing incomplete cache: {final}")
    tmp = final.parent / f".tmp_{final.name}_{backbone.replace('-', '_')}"
    if tmp.exists():
        raise FileExistsError(f"stale temporary directory requires inspection: {tmp}")
    tmp.mkdir(parents=True)
    start_iso, start, d4d_map_cache = iso_now(), time.perf_counter(), {}
    method, checkpoint, predict = build_predictor(backbone, device)
    disparity = np.empty((len(records), CACHE_HEIGHT, CACHE_WIDTH), dtype=np.float16)
    valid = np.empty_like(disparity, dtype=np.uint8)
    runtime_ms: list[float] = []
    native_shape: tuple[int, int] | None = None
    try:
        for index, record in enumerate(records):
            left, right = read_pair(record, d4d_map_cache)
            native_shape = native_shape or left.shape[:2]
            pred, elapsed = predict(left, right)
            if tuple(pred.shape) != tuple(left.shape[:2]):
                raise RuntimeError(f"{backbone} prediction shape {pred.shape} does not match source {left.shape[:2]} for {record.frame_id}")
            disparity[index], valid[index] = resize_to_cache(np.asarray(pred, dtype=np.float32), left.shape[1])
            runtime_ms.append(float(elapsed))
            if (index + 1) % 20 == 0 or index + 1 == len(records):
                print(f"{backbone}/{record.domain}: {index + 1}/{len(records)}", flush=True)

        np.save(tmp / "disparity.npy", disparity)
        np.save(tmp / "valid_mask.npy", valid)
        np.save(tmp / "frame_ids.npy", np.asarray([record.frame_id for record in records], dtype="U256"))
        write_manifest(tmp / "frame_manifest.csv", records)
        h, w = native_shape or (0, 0)
        checkpoint_path = CHECKPOINTS.get(backbone)
        inf_h, inf_w, inf_note = inference_resolution(backbone, h, w)
        metadata = {
            "project": "ARGOS v2", "purpose": "frozen OOD backbone cache for multi-domain study",
            "backbone": backbone, "method": method, "checkpoint": checkpoint,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else "n/a",
            "checkpoint_sha256_partial": partial_hash(checkpoint_path) if checkpoint_path else "n/a",
            "domain": records[0].domain, "frame_count": len(records),
            "source_height": h, "source_width": w, "model_inference_height": inf_h,
            "model_inference_width": inf_w, "model_inference_note": inf_note,
            "cache_height": CACHE_HEIGHT, "cache_width": CACHE_WIDTH,
            "disparity_dtype": "float16", "mask_dtype": "uint8",
            "disparity_units": "pixels_at_cache_resolution", "disparity_convention": "positive_left_disparity",
            "disparity_scale_formula": "d_cache = resize(d_native, (144,180)) * (180 / W_native)",
            "validity": "prediction finite and positive; independent of GT",
            "d4d_rectification": "left/right YAML remap before inference" if records[0].rectified else "prepared rectified inputs",
            "frame_manifest": "frame_manifest.csv", "start_time": start_iso,
            "total_runtime_s": time.perf_counter() - start, "mean_inference_ms": float(np.mean(runtime_ms)),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0,
            "script": str(Path(__file__).resolve()), "git_commit": git_commit(), "completion_status": False,
        }
        (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        checks = validate_cache(tmp, records)
        if not checks.get("passed"):
            raise RuntimeError(f"cache integrity failed: {checks}")
        metadata["completion_status"], metadata["integrity_checks"] = True, checks
        (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (tmp / ".complete").write_text(json.dumps({"validated_at": iso_now(), "checks": checks}) + "\n")
        if final.exists():
            if not force:
                raise FileExistsError(final)
            shutil.rmtree(final)
        tmp.rename(final)
        print(json.dumps({"status": "complete", "path": str(final), "checks": checks}, indent=2))
    except Exception:
        # Retain the incomplete temporary artifact for diagnosis rather than deleting evidence.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("d4d", "servct"), required=True)
    parser.add_argument("--backbone", choices=("RAFT-Stereo", "StereoAnywhere"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="D4D anchors or SERV sequences; smoke only")
    parser.add_argument("--specimens", default="", help="comma-separated D4D specimen IDs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if args.domain == "d4d":
        specimens = {value for value in args.specimens.split(",") if value} or None
        records = d4d_records(args.limit, specimens)
    else:
        records = serv_records(args.limit)
    build(records, backbone=args.backbone, output=args.output, device=device, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
