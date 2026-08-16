#!/usr/bin/env python3
"""V3 frozen closure for the D4D joint unseen-backbone experiment.

Additive only: V1/V2 are retained.  V3 fixes provenance and safety gates before
any result exists, while reusing the already-audited causal H4 execution path.
"""
from __future__ import annotations
import argparse, csv, os, shutil, sys, tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from model_design.comparison import run_joint_d4d_v2 as v2

base, ARGOS, RESULTS = v2.base, v2.ARGOS, v2.RESULTS
PROTOCOL = RESULTS / "protocol"; BUNDLE = PROTOCOL / "joint_d4d_v3_freeze_bundle"
FREEZE, INVENTORY, ATTESTATION = BUNDLE / "freeze.json", BUNDLE / "inventory.json", BUNDLE / "cache_build_retrospective_attestation.json"
OUTPUT, OUTPUT_ATTESTATION = RESULTS / "joint_d4d_v3", RESULTS / "joint_d4d_v3/joint_d4d_v3_attestation.json"
BACKBONES, UNSEEN, EXPECTED = base.BACKBONES, base.UNSEEN, base.EXPECTED
V7_RUN = RESULTS / "canonical_h4_ood_v7/runs/d4d/model_design_comparison_canonical_h4__factory"
V7_REFERENCES = {"v7_d4d_diagnostics": V7_RUN / "d4d_diagnostics.csv", "v7_d4d_run_manifest": V7_RUN / "run_manifest.json", "v7_external_attestation": RESULTS / "canonical_h4_ood_v7/external_ood_attestation.json"}

entry, sha256, atomic_json, atomic_csv, read_json, verify_entries = v2.entry, v2.sha256, v2.atomic_json, v2.atomic_csv, v2.read_json, v2.verify_entries


def source_inputs() -> dict[str, dict[str, str]]:
    """Actual union: V1 runtime closure + V2 omissions + V3/parity sources."""
    values = dict(base._source_inputs())
    values.update(v2.source_inputs())
    values["v3_launcher"] = entry(Path(__file__))
    values.update({name: entry(path) for name, path in V7_REFERENCES.items()})
    return values


def _cache_semantics(inventory: Mapping[str, Any]) -> dict[str, Any]:
    cohort_ids: set[str] = set(); per_session: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in inventory["cohort"]:
        pair = (item["specimen"], item["session"]); frames = set(item["frames_current_to_past"])
        if len(frames) != 4: raise RuntimeError(f"invalid V3 cohort window: {item['anchor_id']}")
        cohort_ids.update(frames); per_session[pair].update(frames)
    if len(cohort_ids) != EXPECTED["frames"] or len(per_session) != EXPECTED["sessions"]:
        raise RuntimeError(f"V3 cohort semantic cardinality mismatch: frames={len(cohort_ids)} sessions={len(per_session)}")
    checks = {}
    for backbone in BACKBONES:
        directory = base._cache_root(backbone); metadata = read_json(directory / "metadata.json")
        if metadata.get("domain") != "D4D" or "YAML remap" not in str(metadata.get("d4d_rectification")):
            raise RuntimeError(f"V3 cache domain/rectification mismatch: {backbone}")
        with (directory / "frame_manifest.csv").open(encoding="utf-8") as stream: rows = {r["frame_id"]: r for r in csv.DictReader(stream)}
        for (specimen, session), ids in per_session.items():
            for frame_id in ids:
                row = rows.get(frame_id)
                if row is None or row.get("domain") != "D4D" or row.get("rectified") != "True" or row.get("specimen") != specimen or row.get("session") != session:
                    raise RuntimeError(f"V3 cache cohort semantics mismatch: {backbone}/{frame_id}")
        checks[backbone] = {"domain": "D4D", "rectified": True, "unique_cohort_frames": len(cohort_ids), "sessions": len(per_session)}
    return checks


def _gates() -> dict[str, str]:
    return {
        "efficacy": "both unseen: session-macro disparity EPE and depth MAE delta<0; paired 10k seed0 CI upper<0; both specimen macro-session deltas<0",
        "tail_vs_raw": "both unseen: session-macro disparity Bad3/P99/InvalidRate and depth BadMM10/P99/InvalidRate refined<=raw",
        "update_safety": "both unseen: HUR/HPlus/NewBad1/3/5 and depth HUR/HPlus/NewBad2/5/10 <= seen-control worst plus fixed tolerance (rates .01; disparity HPlus .02px; depth HPlus 20mm)",
        "identity_clean": "both unseen: clean-good IdentityPreservation at Bad1/3/5 and depth Bad2/5/10 >= max(.95, seen-control minimum-.01)",
        "frame_degradation": "both unseen: FrameDegradation Mean/P95/P99/Worst <= seen-control worst plus fixed tolerance (.02px disparity, 20mm depth)",
        "verdict": "PASS only all gates for both unseen; FAIL if an efficacy mean is non-negative or any tail/update/identity/frame gate fails; otherwise NOT_CONFIRMED. Gates frozen before evaluation; never tuned on unseen results."}


def inventory_payload(attestation: Mapping[str, str] | None=None) -> dict[str, Any]:
    value = v2.inventory_payload(); value["inventory_version"] = 3
    value["cache_provenance"] = {"kind": "RETROSPECTIVE_POST_BUILD_ONLY", "contemporaneous_pre_cache_freeze": False,
                                 "statement": "Cache construction predates this evaluation freeze; this V3 inventory binds only retrospective cache evidence.",
                                 "attestation": dict(attestation or {})}
    value["v3_cache_semantics"] = _cache_semantics(value)
    value["v7_parity_reference"] = {name: entry(path) for name, path in V7_REFERENCES.items()}
    value["safety_gate_protocol"] = _gates()
    return value


def validate_inventory(value: Mapping[str, Any]) -> None:
    v2.validate_inventory({**value, "inventory_version": 2})
    if value.get("inventory_version") != 3 or value.get("v3_cache_semantics") != _cache_semantics(value) or value.get("safety_gate_protocol") != _gates():
        raise RuntimeError("invalid V3 inventory")
    provenance = value.get("cache_provenance", {})
    if provenance.get("kind") != "RETROSPECTIVE_POST_BUILD_ONLY" or provenance.get("contemporaneous_pre_cache_freeze") is not False:
        raise RuntimeError("invalid V3 retrospective cache provenance")
    verify_entries(value.get("v7_parity_reference", {}), label="V3 v7 parity reference")


def retrospective_attestation(inventory: Mapping[str, Any]) -> dict[str, Any]:
    value = v2.retrospective_attestation(inventory)
    return value | {"attestation_version": 2, "v3_unique_cohort_frames": EXPECTED["frames"], "v3_cache_semantics": _cache_semantics(inventory)}


def freeze_payload(inventory_sha: str, attestation_sha: str) -> dict[str, Any]:
    return {"project": "ARGOS v2", "freeze_version": 3, "freeze_id": "joint_d4d_v3", "status": "FROZEN_PRE_RUN", "write_once": True,
            "module": base.MODULE, "immutable_sources_and_checkpoints": source_inputs(), "input_inventory": {"path": str(INVENTORY.resolve()), "sha256": inventory_sha},
            "retrospective_cache_build_attestation": {"path": str(ATTESTATION.resolve()), "sha256": attestation_sha}, "output": str(OUTPUT.resolve()),
            "no_training": True, "no_threshold_tuning": True, "dense_predictions_written": False,
            "atomic_publication": "single same-filesystem directory rename for the complete V3 freeze bundle"}


def write_freeze() -> Path:
    if BUNDLE.exists(): verify_frozen_inputs(); return BUNDLE
    PROTOCOL.mkdir(parents=True, exist_ok=True); stage = Path(tempfile.mkdtemp(prefix=".joint_d4d_v3_bundle-", dir=PROTOCOL))
    try:
        draft = inventory_payload(); attestation = retrospective_attestation(draft); atomic_json(stage / ATTESTATION.name, attestation)
        att_entry = {"path": str(ATTESTATION.resolve()), "sha256": sha256(stage / ATTESTATION.name)}
        inventory = inventory_payload(att_entry); validate_inventory(inventory); atomic_json(stage / INVENTORY.name, inventory)
        atomic_json(stage / FREEZE.name, freeze_payload(sha256(stage / INVENTORY.name), sha256(stage / ATTESTATION.name)))
        os.rename(stage, BUNDLE)
    finally: shutil.rmtree(stage, ignore_errors=True)
    verify_frozen_inputs(); return BUNDLE


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, inventory, attestation = read_json(FREEZE), read_json(INVENTORY), read_json(ATTESTATION)
    if freeze.get("project") != "ARGOS v2" or freeze.get("freeze_version") != 3 or freeze.get("freeze_id") != "joint_d4d_v3" or freeze.get("status") != "FROZEN_PRE_RUN" or freeze.get("output") != str(OUTPUT.resolve()): raise RuntimeError("invalid V3 freeze")
    verify_entries(freeze.get("immutable_sources_and_checkpoints", {}), label="V3 source")
    if freeze.get("input_inventory", {}).get("sha256") != sha256(INVENTORY) or freeze.get("retrospective_cache_build_attestation", {}).get("sha256") != sha256(ATTESTATION): raise RuntimeError("V3 bundle hash mismatch")
    validate_inventory(inventory)
    if inventory["cache_provenance"].get("attestation") != {"path": str(ATTESTATION.resolve()), "sha256": sha256(ATTESTATION)}: raise RuntimeError("V3 inventory does not bind exact retrospective attestation")
    if attestation.get("contemporaneous_pre_cache_freeze") is not False or attestation.get("v3_cache_semantics") != _cache_semantics(inventory): raise RuntimeError("invalid V3 retrospective attestation")
    verify_entries(attestation.get("build_logs", {}), label="V3 cache log")
    for backbone, bundle in attestation.get("unseen_cache_bundles", {}).items(): verify_entries(bundle, label=f"V3 {backbone} cache")
    return freeze, inventory


def _mean(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [float(x[name]) for x in rows if x.get(name) not in (None, "") and np.isfinite(float(x[name]))]
    return float(np.mean(values)) if values else None


def _v3_verdict(output: Path, safety: Mapping[str, Any]) -> dict[str, Any]:
    from model_design.metrics.unified_metrics import paired_bootstrap_ci
    spatial = [x for x in csv.DictReader((output / "per_session_metrics.csv").open()) if x["method"] in {"raw", "refined"}]
    aggregate = safety["aggregate"]; result: dict[str, Any] = {"project": "ARGOS v2", "protocol": "joint_d4d_v3", "frozen_gates": _gates(), "backbones": {}}
    controls = [x for x in aggregate if x["backbone"] not in UNSEEN]
    update = ("disparity_HUR", "disparity_HPlus", "disparity_thresholds_1.0_NewBad", "disparity_thresholds_3.0_NewBad", "disparity_thresholds_5.0_NewBad", "depth_HUR", "depth_HPlus", "depth_thresholds_2.0_NewBad", "depth_thresholds_5.0_NewBad", "depth_thresholds_10.0_NewBad")
    identity = ("disparity_thresholds_1.0_IdentityPreservation", "disparity_thresholds_3.0_IdentityPreservation", "disparity_thresholds_5.0_IdentityPreservation", "depth_thresholds_2.0_IdentityPreservation", "depth_thresholds_5.0_IdentityPreservation", "depth_thresholds_10.0_IdentityPreservation")
    frame = tuple(f"{unit}_FrameDegradation_{stat}" for unit in ("disparity", "depth") for stat in ("Mean", "P95", "P99", "Worst"))
    for backbone in BACKBONES:
        if backbone not in UNSEEN: result["backbones"][backbone] = {"status": "CONTROL"}; continue
        raw = [x for x in spatial if x["backbone"] == backbone and x["method"] == "raw"]; refined = [x for x in spatial if x["backbone"] == backbone and x["method"] == "refined"]
        delta = {"disparity_EPE": _mean(refined, "disparity_EPE") - _mean(raw, "disparity_EPE"), "depth_MAE": _mean(refined, "depth_MAE") - _mean(raw, "depth_MAE")}
        boot = {m: paired_bootstrap_ci({f"{x['specimen']}::{x['session']}": float(x[m]) for x in raw}, {f"{x['specimen']}::{x['session']}": float(x[m]) for x in refined}, n_resamples=10000, seed=0) for m in delta}
        breadth = {s: {m: _mean([x for x in refined if x["specimen"] == s], m) - _mean([x for x in raw if x["specimen"] == s], m) for m in delta} for s in ("specimen_2", "specimen_3")}
        own = next(x for x in aggregate if x["backbone"] == backbone)
        rate = lambda n: .02 if n == "disparity_HPlus" else 20.0 if n == "depth_HPlus" or n.startswith("depth_Frame") else .01
        update_pass = all(float(own.get(n, np.inf)) <= max(float(x.get(n, -np.inf)) for x in controls) + rate(n) for n in update)
        identity_pass = all(float(own.get(n, -np.inf)) >= max(.95, min(float(x.get(n, np.inf)) for x in controls) - .01) for n in identity)
        frame_pass = all(float(own.get(n, np.inf)) <= max(float(x.get(n, -np.inf)) for x in controls) + rate(n) for n in frame)
        tails = {"Bad3": (_mean(refined, "disparity_Bad3"), _mean(raw, "disparity_Bad3")), "P99": (_mean(refined, "disparity_P99"), _mean(raw, "disparity_P99")), "InvalidRate": (_mean(refined, "disparity_InvalidRate"), _mean(raw, "disparity_InvalidRate")), "DepthBad10": (_mean(refined, "depth_BadMM10"), _mean(raw, "depth_BadMM10")), "DepthP99": (_mean(refined, "depth_P99"), _mean(raw, "depth_P99")), "DepthInvalid": (_mean(refined, "depth_InvalidRate"), _mean(raw, "depth_InvalidRate"))}
        tail_pass = all(a <= b for a, b in tails.values()); efficacy = all(v < 0 and boot[m]["ci_upper"] is not None and boot[m]["ci_upper"] < 0 for m, v in delta.items()) and all(x[m] < 0 for x in breadth.values() for m in delta)
        passed = efficacy and tail_pass and update_pass and identity_pass and frame_pass; failed = any(v >= 0 for v in delta.values()) or not (tail_pass and update_pass and identity_pass and frame_pass)
        result["backbones"][backbone] = {"status": "PASS" if passed else "FAIL" if failed else "NOT_CONFIRMED", "macro_session_delta": delta, "bootstrap": boot, "specimen_delta": breadth, "gate_pass": {"efficacy": efficacy, "tail_vs_raw": tail_pass, "update_safety": update_pass, "identity_clean": identity_pass, "frame_degradation": frame_pass}, "tail_refined_vs_raw": tails}
    states = [result["backbones"][x]["status"] for x in UNSEEN]; result["joint_unseen_backbone_and_ood"] = "PASS" if states == ["PASS", "PASS"] else "FAIL" if "FAIL" in states else "NOT_CONFIRMED"
    atomic_json(output / "verdicts.json", result); return result


def evaluate(output: Path, device: str, inventory: Mapping[str, Any]) -> None:
    v2.evaluate(output, device, inventory)
    safety = {name: list(csv.DictReader((output / f"safety_per_{name}.csv").open())) for name in ("anchor", "session", "specimen", "aggregate")}
    _v3_verdict(output, safety)


def output_evidence(root: Path) -> dict[str, Any]:
    # V1's counter/shape/no-dense checks remain mandatory, not just V2 files.
    base._validate_output(root)
    return v2.output_evidence(root)


def run(config: argparse.Namespace) -> Path:
    if config.output.exists(): raise FileExistsError("refusing existing V3 output")
    if config.device != "cuda:0" or not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdecimal(): raise RuntimeError("V3 requires numeric CUDA_VISIBLE_DEVICES and cuda:0")
    before = verify_frozen_inputs()
    if config.output.resolve() != OUTPUT.resolve() or before[0].get("output") != str(config.output.resolve()): raise RuntimeError("V3 output differs from frozen output")
    stage = Path(tempfile.mkdtemp(prefix=f".{config.output.name}.stage-", dir=config.output.parent)); child = stage / "result"; child.mkdir()
    try:
        evaluate(child, config.device, before[1])
        for phase in ("after_inference", "after_metrics"):
            if verify_frozen_inputs() != before: raise RuntimeError(f"V3 TOCTOU mismatch {phase}")
        evidence = output_evidence(child); atomic_json(child / "run_manifest.json", {"project": "ARGOS v2", "status": "COMPLETE", "freeze": entry(FREEZE), "inventory": entry(INVENTORY), "output": str(config.output.resolve()), **evidence})
        atomic_json(child / OUTPUT_ATTESTATION.name, {"project": "ARGOS v2", "status": "COMPLETE_JOINT_D4D_V3", "freeze": entry(FREEZE), "inventory": entry(INVENTORY), "output": str(config.output.resolve()), "output_hashes": base._hash_outputs(child)})
        evidence = output_evidence(child); atomic_json(child / "run_manifest.json", read_json(child / "run_manifest.json") | evidence)
        if verify_frozen_inputs() != before: raise RuntimeError("V3 TOCTOU mismatch after_attestation")
        if config.output.exists(): raise FileExistsError("V3 output appeared before publication")
        os.rename(child, config.output); stage.rmdir()
    except BaseException: raise
    return config.output / OUTPUT_ATTESTATION.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--write-freeze", action="store_true"); parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--device", default="cuda:0"); config = parser.parse_args()
    if config.write_freeze:
        if config.output != OUTPUT: raise ValueError("V3 freeze output is fixed")
        print(write_freeze())
    else: print(run(config))
if __name__ == "__main__": main()
