#!/usr/bin/env python3
"""V5 closure: consume the actual safety-table filenames emitted by V2."""
from __future__ import annotations
import argparse, csv, os, shutil, sys, tempfile
from pathlib import Path
from typing import Any, Mapping
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from model_design.comparison import run_joint_d4d_v4 as v4

base, v2, ARGOS, RESULTS = v4.base, v4.v2, v4.ARGOS, v4.RESULTS
PROTOCOL = RESULTS / "protocol"; BUNDLE = PROTOCOL / "joint_d4d_v5_freeze_bundle"
FREEZE, INVENTORY, ATTESTATION = BUNDLE / "freeze.json", BUNDLE / "inventory.json", BUNDLE / "cache_build_retrospective_attestation.json"
OUTPUT, OUTPUT_ATTESTATION = RESULTS / "joint_d4d_v5", RESULTS / "joint_d4d_v5/joint_d4d_v5_attestation.json"
BACKBONES, UNSEEN, EXPECTED = base.BACKBONES, base.UNSEEN, base.EXPECTED
TABLES = {"anchor": "safety_per_anchor.csv", "session": "safety_per_session.csv", "specimen": "safety_per_specimen.csv", "aggregate": "safety_aggregate.csv"}
entry, sha256, atomic_json, read_json, verify_entries = v4.entry, v4.sha256, v4.atomic_json, v4.read_json, v4.verify_entries


def source_inputs() -> dict[str, dict[str, str]]:
    values = dict(v4.source_inputs()); values["v5_launcher"] = entry(Path(__file__)); return values

def inventory_payload(attestation: Mapping[str, str] | None=None) -> dict[str, Any]:
    value = v4.inventory_payload(attestation); value["inventory_version"] = 5; value["safety_table_contract"] = {"filenames": TABLES, "counts": {"anchor": 336, "session": 80, "specimen": 8, "aggregate": 4}, "aggregate_session_count": 20, "required_columns": ["disparity_HUR", "disparity_HPlus", "disparity_BPlus", "disparity_BUR", "disparity_thresholds_1.0_NewBad", "disparity_thresholds_1.0_IdentityPreservation", "disparity_FrameDegradation_Worst", "depth_HUR", "depth_HPlus", "depth_BPlus", "depth_BUR", "depth_thresholds_2.0_NewBad", "depth_thresholds_2.0_IdentityPreservation", "depth_FrameDegradation_Worst"]}; return value

def validate_inventory(value: Mapping[str, Any]) -> None:
    v4.validate_inventory({**value, "inventory_version": 4})
    contract = value.get("safety_table_contract", {})
    if value.get("inventory_version") != 5 or contract.get("filenames") != TABLES or contract.get("counts") != {"anchor":336,"session":80,"specimen":8,"aggregate":4} or contract.get("aggregate_session_count") != 20 or not contract.get("required_columns"): raise RuntimeError("invalid V5 safety table contract")

def retrospective_attestation(inventory: Mapping[str, Any]) -> dict[str, Any]:
    return v4.retrospective_attestation(inventory) | {"attestation_version": 5}

def freeze_payload(inv_sha: str, att_sha: str) -> dict[str, Any]:
    return {"project":"ARGOS v2","freeze_version":5,"freeze_id":"joint_d4d_v5","status":"FROZEN_PRE_RUN","write_once":True,"module":base.MODULE,"immutable_sources_and_checkpoints":source_inputs(),"input_inventory":{"path":str(INVENTORY.resolve()),"sha256":inv_sha},"retrospective_cache_build_attestation":{"path":str(ATTESTATION.resolve()),"sha256":att_sha},"output":str(OUTPUT.resolve()),"no_training":True,"no_threshold_tuning":True,"dense_predictions_written":False,"atomic_publication":"single same-filesystem directory rename"}

def write_freeze() -> Path:
    if BUNDLE.exists(): verify_frozen_inputs(); return BUNDLE
    PROTOCOL.mkdir(parents=True, exist_ok=True); stage = Path(tempfile.mkdtemp(prefix=".joint_d4d_v5_bundle-", dir=PROTOCOL))
    try:
        draft = inventory_payload(); att = retrospective_attestation(draft); atomic_json(stage / ATTESTATION.name, att); ref = {"path":str(ATTESTATION.resolve()),"sha256":sha256(stage / ATTESTATION.name)}
        inv = inventory_payload(ref); validate_inventory(inv); atomic_json(stage / INVENTORY.name, inv); atomic_json(stage / FREEZE.name, freeze_payload(sha256(stage / INVENTORY.name), sha256(stage / ATTESTATION.name))); os.rename(stage, BUNDLE)
    finally: shutil.rmtree(stage, ignore_errors=True)
    verify_frozen_inputs(); return BUNDLE

def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, inv, att = read_json(FREEZE), read_json(INVENTORY), read_json(ATTESTATION)
    if freeze.get("freeze_version") != 5 or freeze.get("freeze_id") != "joint_d4d_v5" or freeze.get("status") != "FROZEN_PRE_RUN" or freeze.get("output") != str(OUTPUT.resolve()): raise RuntimeError("invalid V5 freeze")
    verify_entries(freeze.get("immutable_sources_and_checkpoints", {}), label="V5 source")
    if freeze.get("input_inventory",{}).get("sha256") != sha256(INVENTORY) or freeze.get("retrospective_cache_build_attestation",{}).get("sha256") != sha256(ATTESTATION): raise RuntimeError("V5 bundle hash mismatch")
    validate_inventory(inv)
    if inv["cache_provenance"].get("attestation") != {"path":str(ATTESTATION.resolve()),"sha256":sha256(ATTESTATION)}: raise RuntimeError("invalid V5 retrospective binding")
    return freeze, inv

def load_safety_tables(output: Path, contract: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {}
    for level, name in contract["filenames"].items():
        path = output / name
        if not path.is_file(): raise RuntimeError(f"missing safety table: {name}")
        with path.open(encoding="utf-8") as stream: rows[level] = list(csv.DictReader(stream))
        if len(rows[level]) != contract["counts"][level]: raise RuntimeError(f"wrong safety row count: {name}")
        fields = set(rows[level][0]) if rows[level] else set()
        missing = set(contract["required_columns"]) - fields
        if missing: raise RuntimeError(f"missing safety columns: {name}/{sorted(missing)}")
        if {row.get("backbone") for row in rows[level]} != set(BACKBONES): raise RuntimeError(f"incomplete safety backbone coverage: {name}")
    if any(str(row.get("session_count")) != str(contract["aggregate_session_count"]) for row in rows["aggregate"]): raise RuntimeError("aggregate safety session count mismatch")
    return rows

def evaluate(output: Path, device: str, inventory: Mapping[str, Any]) -> None:
    # Existing V2 is the producer; do not copy/reimplement causal inference.
    v2.evaluate(output, device, inventory)
    safety = load_safety_tables(output, inventory["safety_table_contract"])
    v4._v4_verdict(output, safety)

def run(config: argparse.Namespace) -> Path:
    if config.output.exists(): raise FileExistsError("refusing existing V5 output")
    if config.device != "cuda:0" or not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdecimal(): raise RuntimeError("V5 requires numeric CUDA_VISIBLE_DEVICES and cuda:0")
    before = verify_frozen_inputs()
    if config.output.resolve() != OUTPUT.resolve() or before[0].get("output") != str(config.output.resolve()): raise RuntimeError("V5 output differs from frozen output")
    stage = Path(tempfile.mkdtemp(prefix=f".{config.output.name}.stage-", dir=config.output.parent)); child = stage / "result"; child.mkdir()
    try:
        evaluate(child, config.device, before[1])
        for phase in ("after_inference","after_metrics"):
            if verify_frozen_inputs() != before: raise RuntimeError(f"V5 TOCTOU mismatch {phase}")
        evidence = v4.v3.output_evidence(child); atomic_json(child / "run_manifest.json", {"project":"ARGOS v2","status":"COMPLETE","freeze":entry(FREEZE),"inventory":entry(INVENTORY),"output":str(config.output.resolve()),**evidence}); atomic_json(child / OUTPUT_ATTESTATION.name, {"project":"ARGOS v2","status":"COMPLETE_JOINT_D4D_V5","freeze":entry(FREEZE),"inventory":entry(INVENTORY),"output":str(config.output.resolve()),"output_hashes":base._hash_outputs(child)}); evidence = v4.v3.output_evidence(child); atomic_json(child / "run_manifest.json", read_json(child / "run_manifest.json") | evidence)
        if verify_frozen_inputs() != before: raise RuntimeError("V5 TOCTOU mismatch after_attestation")
        if config.output.exists(): raise FileExistsError("V5 output appeared before publication")
        os.rename(child, config.output); stage.rmdir()
    except BaseException: raise
    return config.output / OUTPUT_ATTESTATION.name

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--write-freeze",action="store_true"); p.add_argument("--output",type=Path,default=OUTPUT); p.add_argument("--device",default="cuda:0"); c=p.parse_args()
    if c.write_freeze:
        if c.output != OUTPUT: raise ValueError("V5 freeze output is fixed")
        print(write_freeze())
    else: print(run(c))
if __name__ == "__main__": main()
