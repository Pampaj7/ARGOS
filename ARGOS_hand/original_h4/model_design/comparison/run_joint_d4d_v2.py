#!/usr/bin/env python3
"""Additive, write-once V2 closure for joint unseen-backbone + D4D transfer.

V2 leaves all V1 files untouched.  It reuses V1's causal evaluator but records
the complete unified-metric safety report before it is flattened, then applies
the frozen joint efficacy *and* safety gates at session granularity.
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
sys.path.insert(0, str(ROOT))
from model_design.comparison import run_joint_d4d_v1 as base

ARGOS, RESULTS = base.ARGOS, base.RESULTS
PROTOCOL = RESULTS / "protocol"
BUNDLE = PROTOCOL / "joint_d4d_v2_freeze_bundle"
FREEZE, INVENTORY, ATTESTATION = BUNDLE / "freeze.json", BUNDLE / "inventory.json", BUNDLE / "cache_build_retrospective_attestation.json"
OUTPUT = RESULTS / "joint_d4d_v2"
OUTPUT_ATTESTATION = OUTPUT / "joint_d4d_v2_attestation.json"
BACKBONES, UNSEEN, EXPECTED = base.BACKBONES, base.UNSEEN, base.EXPECTED


def entry(path: Path) -> dict[str, str]: return base.entry(path)
def sha256(path: Path) -> str: return base.sha256(path)
def atomic_json(path: Path, value: Mapping[str, Any]) -> None: base.atomic_json(path, value)
def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None: base.atomic_csv(path, rows)
def read_json(path: Path) -> dict[str, Any]: return base.read_json(path)
def verify_entries(values: Mapping[str, Any], *, label: str) -> None: base.verify_entries(values, label=label)


def _logs() -> dict[str, dict[str, str]]:
    root = RESULTS / "joint_shift_cache_logs"
    return {"CREStereo": entry(root / "cres_d4d.log"), "Fast-FoundationStereo": entry(root / "fast_foundation_d4d.log")}


def _inference_manifest_artifacts() -> dict[str, dict[str, str]]:
    manifest = read_json(ROOT / "model_design/checkpoints/inference_manifest.json")
    out = {}
    for name, item in manifest["artifacts"].items():
        path = Path(item["path"])
        out[f"inference_manifest/{name}"] = entry(path if path.is_absolute() else ROOT / path)
    return out


def source_inputs() -> dict[str, dict[str, str]]:
    frozen = ARGOS / "ARGOS_FREEZED/experiments/02_massive_training/scripts/provenance"
    paths = {
        "v2_launcher": Path(__file__), "v1_reused_evaluator": Path(base.__file__),
        "canonical_provenance": ROOT / "scripts/canonical_h4_provenance.py",
        "inference_manifest": ROOT / "model_design/checkpoints/inference_manifest.json",
        "frozen_stereo_photometric": frozen / "stereo_photometric.py",
        "d4d_keyframe_gt": ARGOS / "scripts/temporal_refinement/ood/d4d/d4d_keyframe_gt.py",
        "d4d_cache_builder": ARGOS / "ARGOS-V2/scripts/build_multidomain_backbone_cache.py",
        "joint_cache_builder": ARGOS / "ARGOS-V2/scripts/build_joint_d4d_backbone_cache.py",
        "cache_backbone_registry": ARGOS / "ARGOS-V2/scripts/argos_v2/backbones.py",
        "unified_metrics": ROOT / "model_design/metrics/unified_metrics.py",
        "crestereo_checkpoint": ARGOS / "external/frame_stereo_repos/stereo_matching_crestereo/stereo_matching_crestereo/epoch-570.pth",
        "fast_checkpoint": ARGOS / "external/frame_stereo_repos/Fast-FoundationStereo/weights/onnx/20_30_48/320x736/20_30_48_iters_4_res_320x736.onnx",
    }
    inputs = {name: entry(path) for name, path in paths.items()}
    inputs.update(_inference_manifest_artifacts())
    return inputs


def _validate_cache_semantics(inventory: Mapping[str, Any]) -> dict[str, Any]:
    cohort = inventory["cohort"]
    expected_by_pair = {(item["specimen"], item["session"]): set(item["frames_current_to_past"]) for item in cohort}
    checks: dict[str, Any] = {}
    for backbone in BACKBONES:
        base_dir = base._cache_root(backbone); meta = read_json(base_dir / "metadata.json")
        if meta.get("domain") != "D4D" or "YAML remap" not in str(meta.get("d4d_rectification")):
            raise RuntimeError(f"V2 cache domain/rectification mismatch: {backbone}")
        with (base_dir / "frame_manifest.csv").open(encoding="utf-8") as stream:
            rows = {row["frame_id"]: row for row in csv.DictReader(stream)}
        for (specimen, session), ids in expected_by_pair.items():
            for frame_id in ids:
                row = rows.get(frame_id)
                if row is None or row.get("domain") != "D4D" or row.get("rectified") != "True" or row.get("specimen") != specimen or row.get("session") != session:
                    raise RuntimeError(f"V2 cache cohort semantics mismatch: {backbone}/{frame_id}")
        checks[backbone] = {"domain": meta["domain"], "rectified": meta["d4d_rectification"], "cohort_frame_count": sum(len(x) for x in expected_by_pair.values())}
    return checks


def retrospective_attestation(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Honest post-build evidence; it cannot prove a pre-cache freeze existed."""
    unseen = {backbone: inventory["caches"][backbone] for backbone in UNSEEN}
    metadata = {backbone: read_json(base._cache_root(backbone) / "metadata.json") for backbone in UNSEEN}
    expected = source_inputs()
    checkpoint_hashes = {"CREStereo": expected["crestereo_checkpoint"]["sha256"], "Fast-FoundationStereo": expected["fast_checkpoint"]["sha256"]}
    if {backbone: metadata[backbone].get("checkpoint_sha256") for backbone in UNSEEN} != checkpoint_hashes:
        raise RuntimeError("retrospective cache metadata checkpoint hash mismatch")
    return {"project": "ARGOS v2", "attestation_version": 1, "kind": "RETROSPECTIVE_POST_BUILD_CACHE_ATTESTATION",
            "contemporaneous_pre_cache_freeze": False,
            "claim": "Binds currently observed cache bytes, metadata, logs and backbone checkpoint hashes only; it is not proof that those bytes were frozen before cache construction.",
            "unseen_cache_bundles": unseen, "build_logs": _logs(),
            "metadata_checkpoint_sha256": checkpoint_hashes,
            "cache_semantics": _validate_cache_semantics(inventory), "cache_build_authority": "explicit user authorization after the V1 audit"}


def inventory_payload() -> dict[str, Any]:
    value = base.inventory_payload()
    value.update({"project": "ARGOS v2", "inventory_version": 2, "module": base.MODULE,
                  "v2_cache_semantics": _validate_cache_semantics(value),
                  "safety_gate_protocol": {
                      "unit": "session; 20 equal-weight sessions; no tuning",
                      "efficacy": "both unseen: disparity EPE and depth MAE macro-session delta<0, 95% paired bootstrap upper<0, both specimens<0",
                      "safety": "raw identity reference has zero update harm; HUR/HPlus/NewBad must remain at or below seen-control worst plus fixed tolerance (HUR/NewBad 0.01; HPlus 0.02px), tail Bad/Invalid/P99 must not exceed raw; depth uses the same 0.01 rate tolerance and 20mm HPlus tolerance",
                      "verdict": "PASS requires every efficacy+safety condition for both unseen; FAIL if either efficacy mean>=0 or a named safety metric increases; otherwise NOT_CONFIRMED"}})
    return value


def validate_inventory(value: Mapping[str, Any]) -> None:
    # Use V1 cohort validation, then reject silent cache metadata/semantic drift.
    base.validate_inventory({**value, "inventory_version": 1})
    if value.get("inventory_version") != 2 or not isinstance(value.get("v2_cache_semantics"), Mapping) or not isinstance(value.get("safety_gate_protocol"), Mapping):
        raise RuntimeError("invalid V2 inventory")
    _validate_cache_semantics(value)


def freeze_payload(inventory_sha: str, attestation_sha: str) -> dict[str, Any]:
    return {"project": "ARGOS v2", "freeze_version": 2, "freeze_id": "joint_d4d_v2", "status": "FROZEN_PRE_RUN", "write_once": True,
            "module": base.MODULE, "immutable_sources_and_checkpoints": source_inputs(),
            "input_inventory": {"path": str(INVENTORY.resolve()), "sha256": inventory_sha},
            "retrospective_cache_build_attestation": {"path": str(ATTESTATION.resolve()), "sha256": attestation_sha},
            "output": str(OUTPUT.resolve()), "no_training": True, "no_threshold_tuning": True, "dense_predictions_written": False,
            "atomic_publication": "directory rename of freeze.json, inventory.json and retrospective attestation"}


def write_freeze() -> Path:
    if BUNDLE.exists():
        verify_frozen_inputs(); return BUNDLE
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".joint_d4d_v2_bundle-", dir=PROTOCOL))
    try:
        inventory = stage / "inventory.json"; attestation = stage / "cache_build_retrospective_attestation.json"; freeze = stage / "freeze.json"
        value = inventory_payload(); validate_inventory(value); atomic_json(inventory, value)
        atomic_json(attestation, retrospective_attestation(value))
        atomic_json(freeze, freeze_payload(sha256(inventory), sha256(attestation)))
        os.rename(stage, BUNDLE)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    verify_frozen_inputs(); return BUNDLE


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, inventory, attestation = read_json(FREEZE), read_json(INVENTORY), read_json(ATTESTATION)
    if freeze.get("project") != "ARGOS v2" or freeze.get("freeze_version") != 2 or freeze.get("freeze_id") != "joint_d4d_v2" or freeze.get("status") != "FROZEN_PRE_RUN" or freeze.get("output") != str(OUTPUT.resolve()):
        raise RuntimeError("invalid V2 freeze")
    verify_entries(freeze.get("immutable_sources_and_checkpoints", {}), label="V2 source")
    if freeze.get("input_inventory", {}).get("sha256") != sha256(INVENTORY) or freeze.get("retrospective_cache_build_attestation", {}).get("sha256") != sha256(ATTESTATION):
        raise RuntimeError("V2 bundle hash mismatch")
    validate_inventory(inventory)
    if attestation.get("contemporaneous_pre_cache_freeze") is not False or attestation.get("kind") != "RETROSPECTIVE_POST_BUILD_CACHE_ATTESTATION":
        raise RuntimeError("dishonest V2 cache attestation")
    bundles = attestation.get("unseen_cache_bundles", {})
    if set(bundles) != UNSEEN:
        raise RuntimeError("invalid V2 unseen cache bundle set")
    for backbone, bundle in bundles.items():
        verify_entries(bundle, label=f"V2 {backbone} cache")
    verify_entries(attestation.get("build_logs", {}), label="V2 cache log")
    if attestation.get("cache_semantics") != _validate_cache_semantics(inventory): raise RuntimeError("V2 cache semantics changed")
    return freeze, inventory


def _flatten_safety(value: Mapping[str, Any], prefix: str="") -> dict[str, float]:
    out = {}
    for name, item in value.items():
        key = f"{prefix}{name}"
        if isinstance(item, Mapping):
            if "value" in item and item["value"] is not None: out[key] = float(item["value"])
            out.update(_flatten_safety(item, key + "_"))
    return out


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(x[key]) for x in rows if x.get(key) is not None and np.isfinite(float(x[key]))]
    return float(np.mean(values)) if values else None


def _safety_tables(output: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    anchors = list(csv.DictReader((output / "per_anchor_metrics.csv").open()))
    sessions = [x for x in csv.DictReader((output / "per_session_metrics.csv").open()) if x["method"] == "safety"]
    if len(events) != len(anchors) + len(sessions): raise RuntimeError("V2 safety capture order/count mismatch")
    anchor_rows, session_rows = [], []
    for row, event in zip(anchors, events[:len(anchors)]):
        anchor_rows.append({k: row[k] for k in ("backbone", "specimen", "session", "anchor_id")} | {f"disparity_{k}": v for k, v in _flatten_safety(event["safety_disparity"]).items()} | {f"depth_{k}": v for k, v in _flatten_safety(event["safety_depth"]).items()})
    for row, event in zip(sessions, events[len(anchors):]):
        session_rows.append({k: row[k] for k in ("backbone", "specimen", "session")} | {f"disparity_{k}": v for k, v in _flatten_safety(event["safety_disparity"]).items()} | {f"depth_{k}": v for k, v in _flatten_safety(event["safety_depth"]).items()})
    atomic_csv(output / "safety_per_anchor.csv", anchor_rows); atomic_csv(output / "safety_per_session.csv", session_rows)
    keys = sorted({key for row in session_rows for key in row if key.startswith(("disparity_", "depth_"))})
    specimen_rows, aggregate_rows = [], []
    for backbone in BACKBONES:
        for specimen in ("specimen_2", "specimen_3"):
            subset = [x for x in session_rows if x["backbone"] == backbone and x["specimen"] == specimen]
            specimen_rows.append({"backbone": backbone, "specimen": specimen, "session_count": len(subset)} | {key: _mean(subset, key) for key in keys})
        subset = [x for x in session_rows if x["backbone"] == backbone]
        aggregate_rows.append({"backbone": backbone, "session_count": len(subset)} | {key: _mean(subset, key) for key in keys})
    atomic_csv(output / "safety_per_specimen.csv", specimen_rows); atomic_csv(output / "safety_aggregate.csv", aggregate_rows)
    return {"anchor": anchor_rows, "session": session_rows, "specimen": specimen_rows, "aggregate": aggregate_rows}


def _verdict(output: Path, safety: Mapping[str, Any]) -> dict[str, Any]:
    from model_design.metrics.unified_metrics import paired_bootstrap_ci
    metrics = list(csv.DictReader((output / "per_session_metrics.csv").open()))
    spatial = [x for x in metrics if x["method"] in {"raw", "refined"}]
    result: dict[str, Any] = {"project": "ARGOS v2", "protocol": "joint_d4d_v2", "frozen_gates": read_json(INVENTORY)["safety_gate_protocol"], "backbones": {}}
    control = [x for x in safety["aggregate"] if x["backbone"] not in UNSEEN]
    safety_names = ("disparity_HUR", "disparity_HPlus", "disparity_thresholds_1.0_NewBad", "disparity_thresholds_3.0_NewBad", "disparity_thresholds_5.0_NewBad", "depth_HUR", "depth_HPlus", "depth_thresholds_2.0_NewBad", "depth_thresholds_5.0_NewBad", "depth_thresholds_10.0_NewBad")
    for backbone in BACKBONES:
        raw = [x for x in spatial if x["backbone"] == backbone and x["method"] == "raw"]; refined = [x for x in spatial if x["backbone"] == backbone and x["method"] == "refined"]
        if backbone not in UNSEEN: result["backbones"][backbone] = {"status": "CONTROL"}; continue
        delta = {"disparity_EPE": _mean(refined, "disparity_EPE") - _mean(raw, "disparity_EPE"), "depth_MAE": _mean(refined, "depth_MAE") - _mean(raw, "depth_MAE")}
        bootstrap = {metric: paired_bootstrap_ci({f"{x['specimen']}::{x['session']}": float(x[metric]) for x in raw}, {f"{x['specimen']}::{x['session']}": float(x[metric]) for x in refined}, n_resamples=10000, seed=0) for metric in delta}
        breadth = {specimen: {metric: _mean([x for x in refined if x["specimen"] == specimen], metric) - _mean([x for x in raw if x["specimen"] == specimen], metric) for metric in delta} for specimen in ("specimen_2", "specimen_3")}
        own = next(x for x in safety["aggregate"] if x["backbone"] == backbone)
        tolerance = {name: (20.0 if name == "depth_HPlus" else .02 if name == "disparity_HPlus" else .01) for name in safety_names}
        controls = {name: max(float(x.get(name, -np.inf)) for x in control) for name in safety_names if all(x.get(name) is not None for x in control)}
        tails = {"Bad3": (_mean(refined, "disparity_Bad3"), _mean(raw, "disparity_Bad3")), "P99": (_mean(refined, "disparity_P99"), _mean(raw, "disparity_P99")), "InvalidRate": (_mean(refined, "disparity_InvalidRate"), _mean(raw, "disparity_InvalidRate")), "DepthBad10": (_mean(refined, "depth_BadMM10"), _mean(raw, "depth_BadMM10")), "DepthP99": (_mean(refined, "depth_P99"), _mean(raw, "depth_P99")), "DepthInvalidRate": (_mean(refined, "depth_InvalidRate"), _mean(raw, "depth_InvalidRate"))}
        safety_pass = all(float(own.get(name, np.inf)) <= controls.get(name, np.inf) + tolerance[name] for name in controls) and all(a <= b for a, b in tails.values())
        efficacy_pass = all(value < 0 and bootstrap[key]["ci_upper"] is not None and bootstrap[key]["ci_upper"] < 0 for key, value in delta.items()) and all(item[key] < 0 for item in breadth.values() for key in delta)
        fail = any(x >= 0 for x in delta.values()) or not safety_pass
        result["backbones"][backbone] = {"status": "PASS" if efficacy_pass and safety_pass else "FAIL" if fail else "NOT_CONFIRMED", "macro_session_delta": delta, "bootstrap": bootstrap, "specimen_delta": breadth, "safety_against_controls": {name: {"raw_identity_reference": 0.0, "value": own.get(name), "control_worst_plus_tolerance": controls[name] + tolerance[name]} for name in controls}, "tail_refined_vs_raw": tails}
    states = [result["backbones"][x]["status"] for x in UNSEEN]
    result["joint_unseen_backbone_and_ood"] = "PASS" if states == ["PASS", "PASS"] else "FAIL" if "FAIL" in states else "NOT_CONFIRMED"
    atomic_json(output / "verdicts.json", result); return result


def evaluate(output: Path, device: str, inventory: Mapping[str, Any]) -> None:
    """Capture full safety reports while reusing the frozen V1 causal path."""
    events: list[dict[str, Any]] = []; original = base._score
    def capture(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs); events.append(result); return result
    base._score = capture
    try: base.evaluate(inventory, output, device)
    finally: base._score = original
    safety = _safety_tables(output, events); _verdict(output, safety)


def output_evidence(root: Path) -> dict[str, Any]:
    required = ("d4d_no_reference_diagnostics.csv", "per_anchor_metrics.csv", "per_session_metrics.csv", "per_specimen_metrics.csv", "aggregate_metrics.csv", "safety_per_anchor.csv", "safety_per_session.csv", "safety_per_specimen.csv", "safety_aggregate.csv", "verdicts.json")
    if any(not (root / name).is_file() for name in required): raise RuntimeError("incomplete V2 output")
    hashes = base._hash_outputs(root)
    return {"outputs": sorted(hashes), "output_hashes": hashes, "dense_predictions_written": False}


def run(config: argparse.Namespace) -> Path:
    if config.output.exists(): raise FileExistsError("refusing existing V2 output")
    if config.device != "cuda:0" or not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdecimal(): raise RuntimeError("V2 requires numeric CUDA_VISIBLE_DEVICES and cuda:0")
    before = verify_frozen_inputs()
    stage = Path(tempfile.mkdtemp(prefix=f".{config.output.name}.stage-", dir=config.output.parent)); child = stage / "result"; child.mkdir()
    try:
        evaluate(child, config.device, before[1])
        for phase in ("after_inference", "after_metrics"):
            if verify_frozen_inputs() != before: raise RuntimeError(f"V2 TOCTOU mismatch {phase}")
        evidence = output_evidence(child)
        atomic_json(child / "run_manifest.json", {"project": "ARGOS v2", "status": "COMPLETE", "freeze": entry(FREEZE), "inventory": entry(INVENTORY), "output": str(config.output.resolve()), **evidence})
        atomic_json(child / OUTPUT_ATTESTATION.name, {"project": "ARGOS v2", "status": "COMPLETE_JOINT_D4D_V2", "freeze": entry(FREEZE), "inventory": entry(INVENTORY), "output": str(config.output.resolve()), "output_hashes": base._hash_outputs(child)})
        evidence = output_evidence(child); atomic_json(child / "run_manifest.json", read_json(child / "run_manifest.json") | evidence)
        if verify_frozen_inputs() != before: raise RuntimeError("V2 TOCTOU mismatch after_attestation")
        os.rename(child, config.output); stage.rmdir()
    except BaseException: raise
    return config.output / OUTPUT_ATTESTATION.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--write-freeze", action="store_true"); parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--device", default="cuda:0"); config = parser.parse_args()
    if config.write_freeze:
        if config.output != OUTPUT: raise ValueError("V2 freeze output is fixed")
        print(write_freeze())
    else: print(run(config))


if __name__ == "__main__": main()
