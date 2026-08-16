#!/usr/bin/env python3
"""Immutable V4 closure: V3 provenance with corrected frame-safety tolerance."""
from __future__ import annotations
import argparse, csv, os, shutil, sys, tempfile
from pathlib import Path
from typing import Any, Mapping
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from model_design.comparison import run_joint_d4d_v3 as v3

base, v2, ARGOS, RESULTS = v3.base, v3.v2, v3.ARGOS, v3.RESULTS
PROTOCOL = RESULTS / "protocol"; BUNDLE = PROTOCOL / "joint_d4d_v4_freeze_bundle"
FREEZE, INVENTORY, ATTESTATION = BUNDLE / "freeze.json", BUNDLE / "inventory.json", BUNDLE / "cache_build_retrospective_attestation.json"
OUTPUT, OUTPUT_ATTESTATION = RESULTS / "joint_d4d_v4", RESULTS / "joint_d4d_v4/joint_d4d_v4_attestation.json"
BACKBONES, UNSEEN, EXPECTED = base.BACKBONES, base.UNSEEN, base.EXPECTED
entry, sha256, atomic_json, read_json, verify_entries = v3.entry, v3.sha256, v3.atomic_json, v3.read_json, v3.verify_entries


def source_inputs() -> dict[str, dict[str, str]]:
    values = dict(v3.source_inputs())
    values["v4_launcher"] = entry(Path(__file__))
    values["cache_paths"] = entry(ARGOS / "ARGOS-V2/scripts/argos_v2/paths.py")
    return values


def _gates() -> dict[str, str]:
    gates = dict(v3._gates())
    gates["frame_degradation"] = "both unseen: FrameDegradation Mean/P95/P99/Worst <= seen-control worst plus fixed tolerance (0.02px for every disparity frame statistic; 20mm for every depth frame statistic)"
    return gates


def inventory_payload(attestation: Mapping[str, str] | None=None) -> dict[str, Any]:
    # Start at V1/base, deliberately omit V2's legacy 80-frame semantic record.
    value = base.inventory_payload(); value["inventory_version"] = 4
    value.pop("v2_cache_semantics", None)
    value["cache_provenance"] = {"kind": "RETROSPECTIVE_POST_BUILD_ONLY", "contemporaneous_pre_cache_freeze": False,
                                 "statement": "Cache construction predates V4; this inventory binds retrospective evidence only.", "attestation": dict(attestation or {})}
    value["v4_cache_semantics"] = v3._cache_semantics(value)
    value["v7_parity_reference"] = {name: entry(path) for name, path in v3.V7_REFERENCES.items()}
    value["safety_gate_protocol"] = _gates()
    return value


def validate_inventory(value: Mapping[str, Any]) -> None:
    base.validate_inventory({**value, "inventory_version": 1})
    if value.get("inventory_version") != 4 or "v2_cache_semantics" in value or value.get("v4_cache_semantics") != v3._cache_semantics(value) or value.get("safety_gate_protocol") != _gates(): raise RuntimeError("invalid V4 inventory")
    p = value.get("cache_provenance", {})
    if p.get("kind") != "RETROSPECTIVE_POST_BUILD_ONLY" or p.get("contemporaneous_pre_cache_freeze") is not False: raise RuntimeError("invalid V4 cache provenance")
    verify_entries(value.get("v7_parity_reference", {}), label="V4 v7 parity reference")


def retrospective_attestation(inventory: Mapping[str, Any]) -> dict[str, Any]:
    value = v2.retrospective_attestation(inventory)
    value.pop("cache_semantics", None)
    return value | {"attestation_version": 4, "v4_cache_semantics": v3._cache_semantics(inventory), "contemporaneous_pre_cache_freeze": False}


def freeze_payload(inv_sha: str, att_sha: str) -> dict[str, Any]:
    return {"project": "ARGOS v2", "freeze_version": 4, "freeze_id": "joint_d4d_v4", "status": "FROZEN_PRE_RUN", "write_once": True, "module": base.MODULE,
            "immutable_sources_and_checkpoints": source_inputs(), "input_inventory": {"path": str(INVENTORY.resolve()), "sha256": inv_sha}, "retrospective_cache_build_attestation": {"path": str(ATTESTATION.resolve()), "sha256": att_sha},
            "output": str(OUTPUT.resolve()), "no_training": True, "no_threshold_tuning": True, "dense_predictions_written": False, "atomic_publication": "single same-filesystem directory rename"}


def write_freeze() -> Path:
    if BUNDLE.exists(): verify_frozen_inputs(); return BUNDLE
    PROTOCOL.mkdir(parents=True, exist_ok=True); stage = Path(tempfile.mkdtemp(prefix=".joint_d4d_v4_bundle-", dir=PROTOCOL))
    try:
        draft = inventory_payload(); att = retrospective_attestation(draft); atomic_json(stage / ATTESTATION.name, att)
        att_entry = {"path": str(ATTESTATION.resolve()), "sha256": sha256(stage / ATTESTATION.name)}
        inv = inventory_payload(att_entry); validate_inventory(inv); atomic_json(stage / INVENTORY.name, inv)
        atomic_json(stage / FREEZE.name, freeze_payload(sha256(stage / INVENTORY.name), sha256(stage / ATTESTATION.name)))
        os.rename(stage, BUNDLE)
    finally: shutil.rmtree(stage, ignore_errors=True)
    verify_frozen_inputs(); return BUNDLE


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, inv, att = read_json(FREEZE), read_json(INVENTORY), read_json(ATTESTATION)
    if freeze.get("freeze_version") != 4 or freeze.get("freeze_id") != "joint_d4d_v4" or freeze.get("status") != "FROZEN_PRE_RUN" or freeze.get("output") != str(OUTPUT.resolve()): raise RuntimeError("invalid V4 freeze")
    verify_entries(freeze.get("immutable_sources_and_checkpoints", {}), label="V4 source")
    if freeze.get("input_inventory", {}).get("sha256") != sha256(INVENTORY) or freeze.get("retrospective_cache_build_attestation", {}).get("sha256") != sha256(ATTESTATION): raise RuntimeError("V4 bundle hash mismatch")
    validate_inventory(inv)
    if inv["cache_provenance"].get("attestation") != {"path": str(ATTESTATION.resolve()), "sha256": sha256(ATTESTATION)} or att.get("v4_cache_semantics") != v3._cache_semantics(inv): raise RuntimeError("invalid V4 retrospective binding")
    verify_entries(att.get("build_logs", {}), label="V4 cache log")
    for backbone, bundle in att.get("unseen_cache_bundles", {}).items(): verify_entries(bundle, label=f"V4 {backbone} cache")
    return freeze, inv


def _mean(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [float(x[name]) for x in rows if x.get(name) not in (None, "") and np.isfinite(float(x[name]))]
    return float(np.mean(values)) if values else None


def _v4_verdict(output: Path, safety: Mapping[str, Any]) -> None:
    """V3 verdict with the frozen 0.02px tolerance for all disparity frame stats."""
    from model_design.metrics.unified_metrics import paired_bootstrap_ci
    spatial = [x for x in csv.DictReader((output / "per_session_metrics.csv").open()) if x["method"] in {"raw", "refined"}]; controls = [x for x in safety["aggregate"] if x["backbone"] not in UNSEEN]
    result: dict[str, Any] = {"project": "ARGOS v2", "protocol": "joint_d4d_v4", "frozen_gates": _gates(), "backbones": {}}
    update = ("disparity_HUR", "disparity_HPlus", "disparity_thresholds_1.0_NewBad", "disparity_thresholds_3.0_NewBad", "disparity_thresholds_5.0_NewBad", "depth_HUR", "depth_HPlus", "depth_thresholds_2.0_NewBad", "depth_thresholds_5.0_NewBad", "depth_thresholds_10.0_NewBad")
    identity = ("disparity_thresholds_1.0_IdentityPreservation", "disparity_thresholds_3.0_IdentityPreservation", "disparity_thresholds_5.0_IdentityPreservation", "depth_thresholds_2.0_IdentityPreservation", "depth_thresholds_5.0_IdentityPreservation", "depth_thresholds_10.0_IdentityPreservation")
    frame = tuple(f"{u}_FrameDegradation_{s}" for u in ("disparity", "depth") for s in ("Mean", "P95", "P99", "Worst"))
    for backbone in BACKBONES:
        if backbone not in UNSEEN: result["backbones"][backbone] = {"status": "CONTROL"}; continue
        raw = [x for x in spatial if x["backbone"] == backbone and x["method"] == "raw"]; refined = [x for x in spatial if x["backbone"] == backbone and x["method"] == "refined"]
        delta = {"disparity_EPE": _mean(refined, "disparity_EPE") - _mean(raw, "disparity_EPE"), "depth_MAE": _mean(refined, "depth_MAE") - _mean(raw, "depth_MAE")}
        boot = {m: paired_bootstrap_ci({f"{x['specimen']}::{x['session']}": float(x[m]) for x in raw}, {f"{x['specimen']}::{x['session']}": float(x[m]) for x in refined}, n_resamples=10000, seed=0) for m in delta}
        breadth = {s: {m: _mean([x for x in refined if x["specimen"] == s], m) - _mean([x for x in raw if x["specimen"] == s], m) for m in delta} for s in ("specimen_2", "specimen_3")}; own = next(x for x in safety["aggregate"] if x["backbone"] == backbone)
        tol = lambda n: .02 if n == "disparity_HPlus" or n.startswith("disparity_Frame") else 20.0 if n == "depth_HPlus" or n.startswith("depth_Frame") else .01
        update_ok = all(float(own.get(n, np.inf)) <= max(float(x.get(n, -np.inf)) for x in controls) + tol(n) for n in update); identity_ok = all(float(own.get(n, -np.inf)) >= max(.95, min(float(x.get(n, np.inf)) for x in controls) - .01) for n in identity); frame_ok = all(float(own.get(n, np.inf)) <= max(float(x.get(n, -np.inf)) for x in controls) + tol(n) for n in frame)
        tails = {"Bad3": (_mean(refined, "disparity_Bad3"), _mean(raw, "disparity_Bad3")), "P99": (_mean(refined, "disparity_P99"), _mean(raw, "disparity_P99")), "InvalidRate": (_mean(refined, "disparity_InvalidRate"), _mean(raw, "disparity_InvalidRate")), "DepthBad10": (_mean(refined, "depth_BadMM10"), _mean(raw, "depth_BadMM10")), "DepthP99": (_mean(refined, "depth_P99"), _mean(raw, "depth_P99")), "DepthInvalid": (_mean(refined, "depth_InvalidRate"), _mean(raw, "depth_InvalidRate"))}; tails_ok = all(a <= b for a, b in tails.values()); efficacy = all(v < 0 and boot[m]["ci_upper"] is not None and boot[m]["ci_upper"] < 0 for m, v in delta.items()) and all(x[m] < 0 for x in breadth.values() for m in delta); passed = efficacy and update_ok and identity_ok and frame_ok and tails_ok
        result["backbones"][backbone] = {"status": "PASS" if passed else "FAIL", "macro_session_delta": delta, "bootstrap": boot, "specimen_delta": breadth, "gate_pass": {"efficacy": efficacy, "tail_vs_raw": tails_ok, "update_safety": update_ok, "identity_clean": identity_ok, "frame_degradation": frame_ok}, "tail_refined_vs_raw": tails}
    states = [result["backbones"][x]["status"] for x in UNSEEN]; result["joint_unseen_backbone_and_ood"] = "PASS" if states == ["PASS", "PASS"] else "FAIL" if "FAIL" in states else "NOT_CONFIRMED"; atomic_json(output / "verdicts.json", result)


def evaluate(output: Path, device: str, inventory: Mapping[str, Any]) -> None:
    v3.evaluate(output, device, inventory)
    safety = {name: list(csv.DictReader((output / f"safety_per_{name}.csv").open())) for name in ("anchor", "session", "specimen", "aggregate")}; _v4_verdict(output, safety)


def run(config: argparse.Namespace) -> Path:
    if config.output.exists(): raise FileExistsError("refusing existing V4 output")
    if config.device != "cuda:0" or not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdecimal(): raise RuntimeError("V4 requires numeric CUDA_VISIBLE_DEVICES and cuda:0")
    before = verify_frozen_inputs()
    if config.output.resolve() != OUTPUT.resolve() or before[0].get("output") != str(config.output.resolve()): raise RuntimeError("V4 output differs from frozen output")
    stage = Path(tempfile.mkdtemp(prefix=f".{config.output.name}.stage-", dir=config.output.parent)); child = stage / "result"; child.mkdir()
    try:
        evaluate(child, config.device, before[1])
        for phase in ("after_inference", "after_metrics"):
            if verify_frozen_inputs() != before: raise RuntimeError(f"V4 TOCTOU mismatch {phase}")
        evidence = v3.output_evidence(child); atomic_json(child / "run_manifest.json", {"project": "ARGOS v2", "status": "COMPLETE", "freeze": entry(FREEZE), "inventory": entry(INVENTORY), "output": str(config.output.resolve()), **evidence}); atomic_json(child / OUTPUT_ATTESTATION.name, {"project": "ARGOS v2", "status": "COMPLETE_JOINT_D4D_V4", "freeze": entry(FREEZE), "inventory": entry(INVENTORY), "output": str(config.output.resolve()), "output_hashes": base._hash_outputs(child)}); evidence = v3.output_evidence(child); atomic_json(child / "run_manifest.json", read_json(child / "run_manifest.json") | evidence)
        if verify_frozen_inputs() != before: raise RuntimeError("V4 TOCTOU mismatch after_attestation")
        if config.output.exists(): raise FileExistsError("V4 output appeared before publication")
        os.rename(child, config.output); stage.rmdir()
    except BaseException: raise
    return config.output / OUTPUT_ATTESTATION.name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--write-freeze", action="store_true"); p.add_argument("--output", type=Path, default=OUTPUT); p.add_argument("--device", default="cuda:0"); c = p.parse_args()
    if c.write_freeze:
        if c.output != OUTPUT: raise ValueError("V4 freeze output is fixed")
        print(write_freeze())
    else: print(run(c))
if __name__ == "__main__": main()
