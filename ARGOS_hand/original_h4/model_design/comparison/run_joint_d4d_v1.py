#!/usr/bin/env python3
"""Write-once frozen joint unseen-backbone + D4D evaluation (ARGOS v2).

This is deliberately additive to ``canonical_h4_ood_v7``.  It evaluates the
fixed 84-anchor D4D subset only after the two explicitly authorised unseen
backbone caches exist; GT is loaded strictly after canonical-H4 inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARGOS = ROOT.parents[1]
RESULTS = ROOT.parent / "results/definitive_temporal_evaluation_csv"
PROTOCOL = RESULTS / "protocol"
FREEZE = PROTOCOL / "joint_d4d_v1_freeze.json"
INVENTORY = PROTOCOL / "joint_d4d_v1_input_inventory.json"
OUTPUT = RESULTS / "joint_d4d_v1"
ATTESTATION = OUTPUT / "joint_d4d_v1_attestation.json"
MODULE = "model_design.comparison.canonical_h4:factory"
BACKBONES = ("RAFT-Stereo", "StereoAnywhere", "CREStereo", "Fast-FoundationStereo")
UNSEEN = frozenset(("CREStereo", "Fast-FoundationStereo"))
CACHE_FILES = (".complete", "metadata.json", "frame_manifest.csv", "disparity.npy", "valid_mask.npy", "frame_ids.npy")
NATIVE_HW, CACHE_HW = (714, 894), (144, 180)
EXPECTED = {"gt_anchors": 239, "mapped_contexts": 156, "anchors": 84, "frames": 224, "sessions": 20,
            "diagnostics": 1008, "geometry_rows": 336}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def entry(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        tmp = Path(stream.name); json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        tmp = Path(stream.name); writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def verify_entries(values: Mapping[str, Any], *, label: str) -> None:
    if not values:
        raise RuntimeError(f"empty {label}")
    for name, value in values.items():
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str):
            raise RuntimeError(f"invalid {label}: {name}")
        path = Path(value["path"])
        if not path.is_file() or sha256(path) != value["sha256"]:
            raise RuntimeError(f"{label} hash mismatch: {name}")


def _cache_root(backbone: str) -> Path:
    return ARGOS / "ARGOS-V2/cache_multidomain_backbones" / backbone / "D4D"


def _context_paths() -> tuple[Path, Path, Path]:
    root = ARGOS / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot"
    return root / "context_manifest.csv", root / "d4d_index.csv", ARGOS / "dataset/D4D/processed/keyframe_stereo_gt_curated/manifests/valid_and_warning_manifest.csv"


def _relative(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ARGOS / path


def _cache_manifest(backbone: str) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    base = _cache_root(backbone)
    meta = read_json(base / "metadata.json")
    if (not (base / ".complete").is_file() or not meta.get("completion_status") or
            meta.get("disparity_convention") != "positive_left_disparity" or
            meta.get("disparity_units") != "pixels_at_cache_resolution" or
            (meta.get("cache_height"), meta.get("cache_width")) != CACHE_HW):
        raise RuntimeError(f"incompatible D4D cache: {base}")
    ids = [str(x) for x in np.load(base / "frame_ids.npy", allow_pickle=True).tolist()]
    disp, valid = np.load(base / "disparity.npy", mmap_mode="r"), np.load(base / "valid_mask.npy", mmap_mode="r")
    if (disp.shape != valid.shape or disp.ndim != 3 or tuple(disp.shape[1:]) != CACHE_HW or len(ids) != len(set(ids)) or
            len(ids) != disp.shape[0] or not np.isfinite(np.asarray(disp)).all() or
            not (np.asarray(disp)[np.asarray(valid, bool)] > 0).all()):
        raise RuntimeError(f"invalid D4D cache arrays: {base}")
    with (base / "frame_manifest.csv").open(encoding="utf-8") as stream:
        rows = {row["frame_id"]: row for row in csv.DictReader(stream)}
    if list(rows) != ids:
        raise RuntimeError(f"D4D cache manifest / frame-id mismatch: {base}")
    return rows, meta


def cohort() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the fixed common cohort; only this function can select anchors."""
    context_path, index_path, gt_path = _context_paths()
    with context_path.open(encoding="utf-8") as stream:
        contexts = {row["anchor_id"]: row for row in csv.DictReader(stream)}
    with index_path.open(encoding="utf-8") as stream:
        index = [row for row in csv.DictReader(stream) if row["sequence_id"] in contexts]
    with gt_path.open(encoding="utf-8") as stream:
        all_gt_rows = list(csv.DictReader(stream))
    gt_rows = [row for row in all_gt_rows if row["anchor_id"] in contexts]
    manifests = {backbone: _cache_manifest(backbone)[0] for backbone in BACKBONES}
    selected: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, str]] = {}
    gt_files: dict[str, dict[str, str]] = {}
    cameras: dict[str, dict[str, str]] = {}
    for gt in gt_rows:
        if gt["specimen_id"] not in {"specimen_2", "specimen_3"}:
            continue
        ctx = contexts[gt["anchor_id"]]
        stems = ctx["context_stems"].split(";")
        timestamps = [float(x) for x in ctx["timestamps"].split(";")]
        if len(stems) != 4 or len(set(stems)) != 4 or ctx.get("padding") != "none" or not all(a > b for a, b in zip(timestamps, timestamps[1:])):
            raise RuntimeError(f"non-causal D4D context: {gt['anchor_id']}")
        ids = [f"{gt['specimen_id']}__{gt['session_id']}__{stem}" for stem in stems]
        if any(frame_id not in manifests[backbone] for backbone in BACKBONES for frame_id in ids):
            continue
        first = manifests[BACKBONES[0]]
        if any(manifests[backbone][frame_id][key] != first[frame_id][key]
               for backbone in BACKBONES[1:] for frame_id in ids for key in ("left_path", "right_path")):
            raise RuntimeError(f"source-pair disagreement: {gt['anchor_id']}")
        meta = read_json(_relative(gt["metadata_path"]))
        if Path(meta.get("stereo_frame", "")).stem != stems[0]:
            raise RuntimeError(f"Zivid/current-frame mismatch: {gt['anchor_id']}")
        for frame_id in ids:
            for key in ("left_path", "right_path"):
                path = Path(first[frame_id][key]); source_files.setdefault(str(path.resolve()), entry(path))
        for key in ("gt_depth_path", "gt_disparity_path", "valid_mask_path", "metadata_path"):
            path = _relative(gt[key]); gt_files.setdefault(str(path.resolve()), entry(path))
        camera_dir = ARGOS / "dataset/D4D/raw/extracted" / gt["specimen_id"] / gt["session_id"] / "camera_info"
        for name in ("left.yaml", "right.yaml"):
            path = camera_dir / name; cameras.setdefault(str(path.resolve()), entry(path))
        selected.append({"anchor_id": gt["anchor_id"], "specimen": gt["specimen_id"], "session": gt["session_id"],
                         "anchor_type": gt["anchor_type"], "frames_current_to_past": ids,
                         "timestamps_current_to_past": timestamps, "gt": {key: gt[key] for key in
                         ("gt_depth_path", "gt_disparity_path", "valid_mask_path", "metadata_path", "fx", "baseline_m", "image_width", "image_height")}})
    selected.sort(key=lambda x: x["anchor_id"])
    unique_frames = {frame for item in selected for frame in item["frames_current_to_past"]}
    sessions = {(item["specimen"], item["session"]) for item in selected}
    if len(all_gt_rows) != EXPECTED["gt_anchors"] or len(gt_rows) != EXPECTED["mapped_contexts"] or len(contexts) != EXPECTED["mapped_contexts"] or len(index) != EXPECTED["mapped_contexts"]:
        raise RuntimeError("unexpected validated D4D availability funnel")
    if len(selected) != EXPECTED["anchors"] or len(unique_frames) != EXPECTED["frames"] or len(sessions) != EXPECTED["sessions"] or {x["specimen"] for x in selected} != {"specimen_2", "specimen_3"}:
        raise RuntimeError(f"joint D4D common cohort is not fixed: anchors={len(selected)} frames={len(unique_frames)} sessions={len(sessions)}")
    return selected, {"contexts": entry(context_path), "index": entry(index_path), "gt_manifest": entry(gt_path),
                      "source_files": source_files, "gt_files": gt_files, "camera_yamls": cameras,
                      "availability": {"validated_gt_anchors": len(all_gt_rows), "mapped_contexts": len(contexts),
                                       "strict_common_anchors": len(selected), "unique_frames": len(unique_frames),
                                       "sessions": len(sessions), "specimens": ["specimen_2", "specimen_3"]}}


def _source_inputs() -> dict[str, dict[str, str]]:
    frozen = ARGOS / "ARGOS_FREEZED"
    paths = {"launcher": Path(__file__), "canonical_h4": ROOT / "model_design/comparison/canonical_h4.py",
             "comparison_driver": ROOT / "model_design/comparison/run_comparison.py", "unified_metrics": ROOT / "model_design/metrics/unified_metrics.py",
             "joint_cache_builder": ARGOS / "ARGOS-V2/scripts/build_joint_d4d_backbone_cache.py",
             "cache_builder_base": ARGOS / "ARGOS-V2/scripts/build_multidomain_backbone_cache.py",
             "d4d_flow_adapter": ROOT / "model_design/external_components/bidavideo.py",
             "bida_pull_warp": frozen / "src/argos_freezed/alignment/bida_pull_warp.py",
             "frozen_codd": frozen / "experiments/02_massive_training/scripts/provenance/codd_style_fusion.py",
             "canonical_checkpoint": ROOT / "model_design/checkpoints/codd_style_h4_best_validation.pt",
             "canonical_policy": ROOT / "model_design/checkpoints/codd_style_h4_policy.json",
             "sea_raft_checkpoint": ARGOS / "external/bidavideo/third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-S.pth",
             "crestereo_checkpoint": ARGOS / "external/frame_stereo_repos/stereo_matching_crestereo/stereo_matching_crestereo/epoch-570.pth",
             "fast_checkpoint": ARGOS / "external/frame_stereo_repos/Fast-FoundationStereo/weights/onnx/20_30_48/320x736/20_30_48_iters_4_res_320x736.onnx",
             "fast_config": ARGOS / "external/frame_stereo_repos/Fast-FoundationStereo/weights/onnx/20_30_48/320x736/20_30_48_iters_4_res_320x736.yaml",
             "backbone_registry": ARGOS / "ARGOS-V2/scripts/argos_v2/backbones.py",
             "crestereo_wrapper": ARGOS / "scripts/scared/eval_scared_external_native.py"}
    sea_root = ARGOS / "external/bidavideo/third_party/SEA-RAFT/core"
    paths.update({f"sea_raft/{path.relative_to(sea_root)}": path for path in sea_root.rglob("*.py") if path.is_file()})
    return {name: entry(path) for name, path in paths.items()}


def inventory_payload() -> dict[str, Any]:
    anchors, common = cohort()
    caches = {backbone: {name: entry(_cache_root(backbone) / name) for name in CACHE_FILES} for backbone in BACKBONES}
    return {"project": "ARGOS v2", "inventory_version": 1, "module": MODULE, "backbones": list(BACKBONES),
            "cohort": anchors, "common_inputs": common, "caches": caches,
            "cache_provenance": "post_build_attestation; caches were user-authorised before this evaluation freeze",
            "metric_protocol": {"native_hw": list(NATIVE_HW), "cache_hw": list(CACHE_HW),
                                "native_prediction": "bilinear(cache disparity)*894/180", "native_validity": "nearest(cache raw validity)",
                                "fixed_support": "Zivid GT valid only; never prediction/flow/support intersection",
                                "invalid_penalty_px": 1000.0, "invalid_penalty_mm": 10000.0,
                                "gt_loaded_after_inference": True, "bootstrap": {"unit": "session", "resamples": 10000, "seed": 0, "confidence": .95}},
            "expected": EXPECTED, "gates": {"pass": "both unseen: MAE disparity/depth CI upper<0; both specimens<0; P99/Bad/Invalid non-regression",
                                                 "not_confirmed": "mean improves but breadth/CI/tail gate fails", "fail": "either unseen mean disparity or depth MAE >=0"}}


def validate_inventory(value: Mapping[str, Any]) -> None:
    if value.get("project") != "ARGOS v2" or value.get("inventory_version") != 1 or value.get("module") != MODULE or value.get("backbones") != list(BACKBONES) or value.get("expected") != EXPECTED:
        raise RuntimeError("invalid joint D4D inventory header")
    anchors, common = value.get("cohort"), value.get("common_inputs")
    if not isinstance(anchors, list) or len(anchors) != EXPECTED["anchors"] or not isinstance(common, Mapping) or common.get("availability", {}).get("strict_common_anchors") != EXPECTED["anchors"]:
        raise RuntimeError("invalid fixed joint cohort")
    if len({x.get("anchor_id") for x in anchors}) != EXPECTED["anchors"] or {x.get("specimen") for x in anchors} != {"specimen_2", "specimen_3"}:
        raise RuntimeError("invalid joint anchor identities")
    for item in anchors:
        frames, times = item.get("frames_current_to_past"), item.get("timestamps_current_to_past")
        if not isinstance(frames, list) or len(frames) != 4 or len(set(frames)) != 4 or not isinstance(times, list) or len(times) != 4 or not all(a > b for a, b in zip(times, times[1:])):
            raise RuntimeError("non-causal frozen joint context")
    for name in ("contexts", "index", "gt_manifest"):
        verify_entries({name: common[name]}, label="joint manifest")
    for name in ("source_files", "gt_files", "camera_yamls"):
        verify_entries(common.get(name, {}), label=f"joint {name}")
    for backbone in BACKBONES:
        verify_entries(value.get("caches", {}).get(backbone, {}), label=f"joint {backbone} cache")


def freeze_payload(inventory_sha: str) -> dict[str, Any]:
    return {"project": "ARGOS v2", "freeze_version": 1, "freeze_id": "joint_d4d_v1", "status": "FROZEN_PRE_RUN", "write_once": True,
            "module": MODULE, "immutable_sources_and_checkpoints": _source_inputs(), "input_inventory": {"path": str(INVENTORY.resolve()), "sha256": inventory_sha},
            "output": str(OUTPUT.resolve()), "no_training": True, "no_threshold_tuning": True, "dense_predictions_written": False}


def write_freeze() -> tuple[Path, Path]:
    if FREEZE.exists() or INVENTORY.exists():
        if FREEZE.exists() and INVENTORY.exists():
            verify_frozen_inputs(); return FREEZE, INVENTORY
        raise RuntimeError("incomplete joint freeze publication")
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".joint_d4d_v1_freeze-", dir=PROTOCOL))
    try:
        inventory = stage / INVENTORY.name; freeze = stage / FREEZE.name
        payload = inventory_payload(); validate_inventory(payload); atomic_json(inventory, payload)
        atomic_json(freeze, freeze_payload(sha256(inventory)))
        os.replace(inventory, INVENTORY); os.replace(freeze, FREEZE)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    verify_frozen_inputs(); return FREEZE, INVENTORY


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, inventory = read_json(FREEZE), read_json(INVENTORY)
    if (freeze.get("project") != "ARGOS v2" or freeze.get("freeze_version") != 1 or freeze.get("freeze_id") != "joint_d4d_v1" or
            freeze.get("status") != "FROZEN_PRE_RUN" or freeze.get("module") != MODULE or freeze.get("output") != str(OUTPUT.resolve()) or
            freeze.get("no_training") is not True or freeze.get("no_threshold_tuning") is not True):
        raise RuntimeError("invalid joint D4D freeze")
    verify_entries(freeze.get("immutable_sources_and_checkpoints", {}), label="joint source")
    pin = freeze.get("input_inventory", {})
    if pin.get("path") != str(INVENTORY.resolve()) or pin.get("sha256") != sha256(INVENTORY):
        raise RuntimeError("joint inventory is not pinned")
    validate_inventory(inventory)
    return freeze, inventory


def _resize_native(tensor: Any, *, mode: str) -> Any:
    import torch.nn.functional as F
    return F.interpolate(tensor, size=NATIVE_HW, mode=mode, align_corners=False if mode == "bilinear" else None)


def _native_prediction(cache: Any, valid: Any, index: int, refined: Any | None=None) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    if refined is None:
        prediction = cv2.resize(np.asarray(cache[index], np.float32), NATIVE_HW[::-1], interpolation=cv2.INTER_LINEAR) * (NATIVE_HW[1] / CACHE_HW[1])
    else:
        prediction = _resize_native(refined.float(), mode="bilinear")[0, 0].detach().cpu().numpy() * (NATIVE_HW[1] / CACHE_HW[1])
    mask = cv2.resize(np.asarray(valid[index], np.uint8), NATIVE_HW[::-1], interpolation=cv2.INTER_NEAREST) > 0
    prediction[~mask] = np.nan
    return prediction, mask


def _plain(summary: Mapping[str, Any]) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    for name, value in summary.items():
        if isinstance(value, Mapping) and "value" in value:
            out[name] = value["value"]
    return out


def _score(raw: np.ndarray, refined: np.ndarray, gt_disp: np.ndarray, raw_depth: np.ndarray, refined_depth: np.ndarray, gt_depth: np.ndarray, support: np.ndarray, config: Any) -> dict[str, Any]:
    from model_design.metrics.unified_metrics import compute_refinement_safety, compute_spatial_metrics
    args = (support[None, None], support[None, None], config)
    raw_spatial = compute_spatial_metrics(raw[None, None], gt_disp[None, None], *args, prediction_depth_mm=raw_depth[None, None], target_depth_mm=gt_depth[None, None])
    refined_spatial = compute_spatial_metrics(refined[None, None], gt_disp[None, None], *args, prediction_depth_mm=refined_depth[None, None], target_depth_mm=gt_depth[None, None])
    return {"raw": {"disparity": _plain(raw_spatial["disparity_px"]["prediction"]), "depth": _plain(raw_spatial["depth_mm"]["prediction"])},
            "refined": {"disparity": _plain(refined_spatial["disparity_px"]["prediction"]), "depth": _plain(refined_spatial["depth_mm"]["prediction"])},
            "safety_disparity": compute_refinement_safety(raw[None, None], refined[None, None], gt_disp[None, None], *args, unit="px"),
            "safety_depth": compute_refinement_safety(raw_depth[None, None], refined_depth[None, None], gt_depth[None, None], *args, unit="mm")}


def _safety_plain(value: Mapping[str, Any]) -> dict[str, float | int | None]:
    output = {name: item.get("value") for name, item in value.items() if isinstance(item, Mapping) and "value" in item}
    for threshold, item in value.get("thresholds", {}).items():
        output.update({f"{name}_{threshold}": detail.get("value") for name, detail in item.items() if isinstance(detail, Mapping)})
    return output


def _load_gt(item: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Called only after H4 returns its final prediction."""
    gt = item["gt"]
    disparity = np.load(_relative(gt["gt_disparity_path"])).astype(np.float32)
    depth = np.load(_relative(gt["gt_depth_path"])).astype(np.float32) * 1000.0
    import cv2
    valid = cv2.imread(str(_relative(gt["valid_mask_path"])), cv2.IMREAD_GRAYSCALE) > 0
    support = valid & np.isfinite(disparity) & (disparity > 0) & np.isfinite(depth) & (depth > 0)
    if disparity.shape != NATIVE_HW or depth.shape != NATIVE_HW or support.shape != NATIVE_HW or not support.any():
        raise RuntimeError(f"invalid/nonempty fixed GT support: {item['anchor_id']}")
    return disparity, depth, support, float(gt["fx"]), float(gt["baseline_m"]) * 1000.0


def _diagnostic(row: Mapping[str, Any], result: Mapping[str, Any], current: Mapping[str, Any], previous: Mapping[str, Any], flow: Any) -> dict[str, Any]:
    from model_design.external_components.bidavideo import temporal_disparity_evidence
    forward, backward = flow(current, previous)
    evidence = temporal_disparity_evidence(current["raw"], previous["raw"], forward, backward, current_valid=current["raw_valid"], past_valid=previous["raw_valid"], current_rgb=current["rgb"], past_rgb=previous["rgb"])
    support = result["support"] & evidence.aligned_validity & evidence.warp_support
    def mean(value: Any) -> float | None:
        return float(value[support].mean()) if bool(support.any()) else None
    diag = result["diagnostics"]
    return {"dataset": "D4D", "metric_scope": "no_reference_prediction_space", "backbone": row["backbone"], "specimen": row["specimen"], "session": row["session"], "anchor_id": row["anchor_id"], "frame_id": current["frame_id"], "step_since_reset": row["step"], "reset": int(result["reset"]),
            "support_coverage": float(support.float().mean()), "raw_mc_inconsistency": mean((current["raw"] - evidence.aligned_past_disparity).abs()),
            "temporal_module_mc_inconsistency": mean((result["disparity"] - result.get("aligned_memory", evidence.aligned_past_disparity)).abs()),
            "update_magnitude": diag.get("update_magnitude"), "temporal_weight": diag.get("temporal_weight"), "fb_confidence": diag.get("fb_confidence")}


def evaluate(inventory: Mapping[str, Any], output: Path, device: str) -> None:
    """Evaluate H4 only; no GT/calibration reaches frames passed to drive()."""
    import cv2
    import torch
    from model_design.comparison.canonical_h4 import factory
    from model_design.comparison.run_comparison import drive
    from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter
    from model_design.metrics.unified_metrics import MetricConfig, disparity_to_depth
    v2 = ARGOS / "ARGOS-V2/scripts"
    if str(v2) not in sys.path: sys.path.insert(0, str(v2))
    from build_multidomain_backbone_cache import d4d_records, read_pair

    caches: dict[str, tuple[Any, Any, dict[str, int]]] = {}
    for backbone in BACKBONES:
        base = _cache_root(backbone); ids = [str(x) for x in np.load(base / "frame_ids.npy", allow_pickle=True).tolist()]
        caches[backbone] = (np.load(base / "disparity.npy", mmap_mode="r"), np.load(base / "valid_mask.npy", mmap_mode="r"), {x: i for i, x in enumerate(ids)})
    source = {record.frame_id: record for record in d4d_records(specimens={"specimen_2", "specimen_3"})}
    adapter, flow_model = factory(device=device), BiDAFlowInferenceAdapter("sea_raft", device=torch.device(device))
    diagnostics: list[dict[str, Any]] = []; anchors: list[dict[str, Any]] = []
    session_samples: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    maps: dict[Any, Any] = {}
    def rgb(value: np.ndarray) -> Any:
        value = cv2.resize(value, (CACHE_HW[1], CACHE_HW[0]), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)[None].float().to(device)
    for backbone in BACKBONES:
        disparity, valid, lookup = caches[backbone]
        for item in inventory["cohort"]:
            ids = list(reversed(item["frames_current_to_past"]))
            frames = []
            for frame_id in ids:
                index = lookup[frame_id]; left, right = read_pair(source[frame_id], maps)
                raw = np.asarray(disparity[index], np.float32); mask = np.asarray(valid[index], bool) & np.isfinite(raw) & (raw > 0)
                frames.append({"index": frame_id, "frame_id": frame_id, "raw": torch.from_numpy(raw)[None, None].to(device), "raw_valid": torch.from_numpy(mask)[None, None].to(device), "rgb": rgb(left), "right_rgb": rgb(right)})
            def flow(current: Mapping[str, Any], past: Mapping[str, Any]):
                return flow_model.current_to_past(current["rgb"], past["rgb"]), flow_model.past_to_current(past["rgb"], current["rgb"])
            outputs = dict(drive(adapter, frames, flow, horizon=4))
            final = outputs[3]
            if final["reset"] or final["state_age"] != 3:
                raise RuntimeError(f"invalid H4 final state: {item['anchor_id']}")
            for step, result in list(outputs.items())[1:]:
                diagnostics.append(_diagnostic({"backbone": backbone, "specimen": item["specimen"], "session": item["session"], "anchor_id": item["anchor_id"], "step": step}, result, frames[step], frames[step - 1], flow))
            # GT starts here: final prediction already exists, and only these local metric values see it.
            gt_disp, gt_depth, support, fx, baseline = _load_gt(item)
            raw, raw_valid = _native_prediction(disparity, valid, lookup[ids[-1]])
            refined, refined_valid = _native_prediction(disparity, valid, lookup[ids[-1]], final["disparity"])
            refined_valid &= np.isfinite(refined) & (refined > 0); refined[~refined_valid] = np.nan
            raw_depth, refined_depth = disparity_to_depth(raw, fx, baseline), disparity_to_depth(refined, fx, baseline)
            score = _score(raw, refined, gt_disp, raw_depth, refined_depth, gt_depth, support, MetricConfig(fx_px=fx, baseline_mm=baseline))
            common = {"dataset": "D4D", "metric_scope": "sparse_zivid_anchor_gt", "protocol": "joint_d4d_v1_fixed_gt_common_support", "backbone": backbone,
                      "backbone_status": "unseen" if backbone in UNSEEN else "seen", "specimen": item["specimen"], "session": item["session"], "anchor_id": item["anchor_id"],
                      "anchor_type": item["anchor_type"], "support_count": int(support.sum()), "gt_coverage": float(support.mean()), "raw_valid_coverage": float(raw_valid.mean()), "refined_valid_coverage": float(refined_valid.mean())}
            anchors.append(common |
                {f"raw_disparity_{k}": v for k, v in score["raw"]["disparity"].items()} |
                {f"raw_depth_{k}": v for k, v in score["raw"]["depth"].items()} |
                {f"refined_disparity_{k}": v for k, v in score["refined"]["disparity"].items()} |
                {f"refined_depth_{k}": v for k, v in score["refined"]["depth"].items()} |
                {f"safety_disparity_{k}": v for k, v in _safety_plain(score["safety_disparity"]).items()} |
                {f"safety_depth_{k}": v for k, v in _safety_plain(score["safety_depth"]).items()})
            key = (backbone, item["specimen"], item["session"])
            for name, value in (("raw_disp", raw), ("refined_disp", refined), ("gt_disp", gt_disp), ("raw_depth", raw_depth), ("refined_depth", refined_depth), ("gt_depth", gt_depth)):
                session_samples[key][name].append(value[support])
    if len(diagnostics) != EXPECTED["diagnostics"] or len(anchors) != EXPECTED["geometry_rows"]:
        raise RuntimeError(f"unexpected joint output counts: diagnostics={len(diagnostics)} anchors={len(anchors)}")
    sessions: list[dict[str, Any]] = []
    for (backbone, specimen, session), values in sorted(session_samples.items()):
        arrays = {name: np.concatenate(parts) for name, parts in values.items()}
        support = np.ones(arrays["gt_disp"].shape, bool); config = MetricConfig()
        score = _score(arrays["raw_disp"], arrays["refined_disp"], arrays["gt_disp"], arrays["raw_depth"], arrays["refined_depth"], arrays["gt_depth"], support, config)
        base = {"dataset": "D4D", "metric_scope": "sparse_zivid_anchor_gt", "backbone": backbone, "backbone_status": "unseen" if backbone in UNSEEN else "seen", "specimen": specimen, "session": session, "anchor_count": sum(1 for x in inventory["cohort"] if x["specimen"] == specimen and x["session"] == session), "support_count": int(support.sum())}
        for method in ("raw", "refined"):
            sessions.append(base | {"method": method} | {f"disparity_{k}": v for k, v in score[method]["disparity"].items()} | {f"depth_{k}": v for k, v in score[method]["depth"].items()})
        sessions.append(base | {"method": "safety"} | {f"disparity_{k}": v for k, v in _safety_plain(score["safety_disparity"]).items()} | {f"depth_{k}": v for k, v in _safety_plain(score["safety_depth"]).items()})
    if len(sessions) != EXPECTED["sessions"] * len(BACKBONES) * 3:
        raise RuntimeError("unexpected session metric count")
    atomic_csv(output / "d4d_no_reference_diagnostics.csv", diagnostics)
    atomic_csv(output / "per_anchor_metrics.csv", anchors)
    atomic_csv(output / "per_session_metrics.csv", sessions)
    _compile(output, sessions, diagnostics)
    _check_v7_parity(diagnostics)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else None


def _compile(output: Path, sessions: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> None:
    from model_design.metrics.unified_metrics import paired_bootstrap_ci
    aggregate: list[dict[str, Any]] = []; specimen_rows: list[dict[str, Any]] = []
    verdicts: dict[str, Any] = {"project": "ARGOS v2", "protocol": "joint_d4d_v1", "backbones": {}}
    metrics = ("disparity_EPE", "disparity_P99", "disparity_Bad3", "disparity_InvalidRate", "depth_MAE", "depth_P99", "depth_BadMM10", "depth_InvalidRate")
    for backbone in BACKBONES:
        rows = [x for x in sessions if x["backbone"] == backbone and x["method"] in {"raw", "refined"}]
        # Keep session pooling primary; no pixel-level pseudo-replication.
        raw = [x for x in rows if x["method"] == "raw"]; refined = [x for x in rows if x["method"] == "refined"]
        if len(raw) != EXPECTED["sessions"] or len(refined) != EXPECTED["sessions"]:
            raise RuntimeError(f"incomplete sessions for {backbone}")
        by_method = {"raw": raw, "refined": refined}
        for specimen in ("specimen_2", "specimen_3"):
            for method, values in by_method.items():
                subset = [x for x in values if x["specimen"] == specimen]
                for metric in metrics:
                    specimen_rows.append({"dataset": "D4D", "scope": "macro_session", "backbone": backbone,
                                          "backbone_status": "unseen" if backbone in UNSEEN else "seen", "specimen": specimen,
                                          "method": method, "metric": metric, "value": _mean(subset, metric), "session_count": len(subset)})
        for method, values in by_method.items():
            for metric in metrics:
                aggregate.append({"dataset": "D4D", "scope": "macro_session", "backbone": backbone, "backbone_status": "unseen" if backbone in UNSEEN else "seen", "method": method, "metric": metric, "value": _mean(values, metric), "session_count": len(values)})
        gates: dict[str, Any] = {"status": "CONTROL" if backbone not in UNSEEN else "NOT_CONFIRMED"}
        if backbone in UNSEEN:
            ci = {}
            for metric in ("disparity_EPE", "depth_MAE"):
                base = {f"{x['specimen']}::{x['session']}": float(x[metric]) for x in raw}
                cand = {f"{x['specimen']}::{x['session']}": float(x[metric]) for x in refined}
                ci[metric] = paired_bootstrap_ci(base, cand, n_resamples=10000, seed=0)
            specimen = {name: {} for name in ("specimen_2", "specimen_3")}
            for name in specimen:
                for metric in ("disparity_EPE", "depth_MAE"):
                    specimen[name][metric] = _mean([x for x in refined if x["specimen"] == name], metric) - _mean([x for x in raw if x["specimen"] == name], metric)
            tail = {metric: (_mean(refined, metric), _mean(raw, metric)) for metric in ("disparity_P99", "disparity_Bad3", "disparity_InvalidRate", "depth_P99", "depth_BadMM10", "depth_InvalidRate")}
            means = {metric: _mean(refined, metric) - _mean(raw, metric) for metric in ("disparity_EPE", "depth_MAE")}
            pass_all = all(means[x] < 0 and ci[x]["ci_upper"] is not None and ci[x]["ci_upper"] < 0 for x in means) and all(v[m] < 0 for v in specimen.values() for m in means) and all(a <= b for a, b in tail.values())
            fail = any(means[x] >= 0 for x in means)
            gates = {"status": "PASS" if pass_all else "FAIL" if fail else "NOT_CONFIRMED", "macro_session_delta": means, "bootstrap": ci, "specimen_macro_session_delta": specimen, "tail_refined_vs_raw": tail,
                     "criteria": "both units mean+CI+specimen breadth negative; P99/Bad/Invalid non-regression"}
        verdicts["backbones"][backbone] = gates
    unseen = [verdicts["backbones"][x]["status"] for x in UNSEEN]
    verdicts["joint_unseen_backbone_and_ood"] = "PASS" if unseen == ["PASS", "PASS"] else "FAIL" if "FAIL" in unseen else "NOT_CONFIRMED"
    for backbone in BACKBONES:
        rows = [x for x in diagnostics if x["backbone"] == backbone]
        aggregate.append({"dataset": "D4D", "scope": "no_reference_transition", "backbone": backbone, "backbone_status": "unseen" if backbone in UNSEEN else "seen", "method": "raw", "metric": "motion_compensated_inconsistency", "value": _mean(rows, "raw_mc_inconsistency"), "session_count": EXPECTED["sessions"]})
        aggregate.append({"dataset": "D4D", "scope": "no_reference_transition", "backbone": backbone, "backbone_status": "unseen" if backbone in UNSEEN else "seen", "method": "refined", "metric": "motion_compensated_inconsistency", "value": _mean(rows, "temporal_module_mc_inconsistency"), "session_count": EXPECTED["sessions"]})
    atomic_csv(output / "per_specimen_metrics.csv", specimen_rows)
    atomic_csv(output / "aggregate_metrics.csv", aggregate); atomic_json(output / "verdicts.json", verdicts)


def _check_v7_parity(rows: list[dict[str, Any]]) -> None:
    path = RESULTS / "canonical_h4_ood_v7/runs/d4d/model_design_comparison_canonical_h4__factory/d4d_diagnostics.csv"
    if not path.is_file():
        raise RuntimeError("missing v7 D4D diagnostic parity reference")
    old = {(x["backbone"], x["sequence"], x["frame_id"], x["step_since_reset"]): x for x in csv.DictReader(path.open(encoding="utf-8"))}
    relevant = [x for x in rows if x["backbone"] in {"RAFT-Stereo", "StereoAnywhere"}]
    if len(relevant) != 504:
        raise RuntimeError("incorrect v7 parity row count")
    for item in relevant:
        key = (item["backbone"], item["anchor_id"], item["frame_id"], str(item["step_since_reset"]))
        if key not in old:
            raise RuntimeError(f"missing v7 parity row: {key}")
        for name in ("support_coverage", "raw_mc_inconsistency", "temporal_module_mc_inconsistency", "update_magnitude"):
            a, b = item[name], old[key][name]
            if a is None or b == "" or not np.isclose(float(a), float(b), atol=1e-6, rtol=1e-5):
                raise RuntimeError(f"v7 diagnostic parity failed: {key}/{name}")


def _hash_outputs(root: Path) -> dict[str, str]:
    files = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name != "run_manifest.json")
    if any(Path(name).suffix in {".npy", ".npz", ".pt", ".pth", ".pkl"} for name in files):
        raise RuntimeError("dense prediction output prohibited")
    return {name: sha256(root / name) for name in files}


def _validate_output(root: Path) -> dict[str, Any]:
    required = ("d4d_no_reference_diagnostics.csv", "per_anchor_metrics.csv", "per_session_metrics.csv", "per_specimen_metrics.csv", "aggregate_metrics.csv", "verdicts.json")
    if any(not (root / name).is_file() for name in required): raise RuntimeError("incomplete joint output")
    if len(list(csv.DictReader((root / "d4d_no_reference_diagnostics.csv").open()))) != EXPECTED["diagnostics"]: raise RuntimeError("incorrect diagnostics output")
    if len(list(csv.DictReader((root / "per_anchor_metrics.csv").open()))) != EXPECTED["geometry_rows"]: raise RuntimeError("incorrect anchor output")
    if len(list(csv.DictReader((root / "per_session_metrics.csv").open()))) != EXPECTED["sessions"] * len(BACKBONES) * 3: raise RuntimeError("incorrect session output")
    hashes = _hash_outputs(root)
    return {"outputs": sorted(hashes), "output_hashes": hashes, "dense_predictions_written": False}


def run(config: argparse.Namespace) -> Path:
    if config.output.exists(): raise FileExistsError(f"refusing existing output: {config.output}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if config.device != "cuda:0" or visible is None or not visible.isdecimal():
        raise RuntimeError("joint evaluation requires one numeric CUDA_VISIBLE_DEVICES and logical --device cuda:0")
    freeze_before, inventory_before = verify_frozen_inputs()
    if config.output.resolve() != OUTPUT.resolve() or freeze_before.get("output") != str(config.output.resolve()): raise RuntimeError("output differs from freeze")
    stage = Path(tempfile.mkdtemp(prefix=f".{config.output.name}.stage-", dir=config.output.parent)); child = stage / "result"
    child.mkdir()
    try:
        evaluate(inventory_before, child, config.device)
        for phase in ("after_inference", "after_compilation"):
            if verify_frozen_inputs() != (freeze_before, inventory_before): raise RuntimeError(f"TOCTOU mismatch {phase}")
        evidence = _validate_output(child)
        atomic_json(child / "run_manifest.json", {"project": "ARGOS v2", "status": "COMPLETE", "freeze": entry(FREEZE), "inventory": entry(INVENTORY), "output": str(config.output.resolve()), **evidence})
        if verify_frozen_inputs() != (freeze_before, inventory_before): raise RuntimeError("TOCTOU mismatch after_manifest")
        atomic_json(child / ATTESTATION.name, {"project": "ARGOS v2", "status": "COMPLETE_JOINT_D4D", "freeze": entry(FREEZE), "inventory": entry(INVENTORY), "output": str(config.output.resolve()), "output_hashes": _hash_outputs(child)})
        evidence = _validate_output(child); manifest = read_json(child / "run_manifest.json")
        atomic_json(child / "run_manifest.json", manifest | evidence)
        if verify_frozen_inputs() != (freeze_before, inventory_before): raise RuntimeError("TOCTOU mismatch after_attestation")
        if config.output.exists(): raise FileExistsError("output appeared during staging")
        os.rename(child, config.output); stage.rmdir()
    except BaseException:
        # Retain the stage for forensic inspection; publication remains absent.
        raise
    return config.output / ATTESTATION.name


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-freeze", action="store_true"); parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    config = arguments()
    if config.write_freeze:
        if config.output != OUTPUT: raise ValueError("freeze path is fixed")
        print("\n".join(map(str, write_freeze())))
    else:
        print(run(config))


if __name__ == "__main__":
    main()
