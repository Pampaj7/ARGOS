#!/usr/bin/env python3
"""Freeze a D2-selected ARGOS v2 recipe; refuses incomplete campaigns."""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from campaign_common import *
def main():
    verify_frozen_core(); manifests=[]
    for budget in (1,3,6):
        for seed in SEEDS:
            path=run_directory(budget,seed)/"manifest.json"
            if not path.is_file(): raise SystemExit(f"incomplete run: {budget}x/{seed}")
            value=json.loads(path.read_text())
            if value.get("exit_status")!="complete": raise SystemExit(f"run not complete: {path}")
            manifests.append(value)
    integrity_path=CAMPAIGN/"aggregate/run_integrity.json"
    if not integrity_path.is_file() or not json.loads(integrity_path.read_text()).get("integrity_pass"):
        raise SystemExit("run integrity missing or failed")
    selection_path=CAMPAIGN/"selection/validation_selection_results.json"
    if not selection_path.is_file(): raise SystemExit("D2 selection results missing")
    selection=json.loads(selection_path.read_text())
    if selection.get("verdict")!="ELIGIBLE BUDGET SELECTED": raise SystemExit("no eligible D2 budget; D7 remains locked")
    budget=int(str(selection["selected_budget"]).removesuffix("x")); checkpoints=[]
    for seed in SEEDS:
        path=run_directory(budget,seed)/"checkpoints/best_validation.pt"; checkpoints.append({"seed":seed,"path":str(path),"sha256":sha256(path)})
    frozen={"project":"ARGOS v2","selected_budget":f"{budget}x","selected_seeds":list(SEEDS),"selected_checkpoints":checkpoints,
            "selection_rule_evaluation":selection,"source_hashes":recipe_hashes(),"frozen_core_hash":FROZEN_MANIFEST_SHA,"data_manifest_hash":sha256(CAMPAIGN/"data_manifest.json"),
            "run_integrity_sha256":sha256(integrity_path),"selection_sha256":sha256(selection_path),"timestamp":datetime.now(timezone.utc).isoformat(),"dataset_7_used":False,"statement":"Dataset 7 was not used for training, validation, budget selection, or checkpoint selection."}
    path=CAMPAIGN/"selection/frozen_recipe_manifest.json"; atomic_json(path,frozen)
    (CAMPAIGN/"selection/frozen_recipe_manifest.sha256").write_text(f"{sha256(path)}  frozen_recipe_manifest.json\n")
    (CAMPAIGN/"selection/selected_checkpoints.sha256").write_text("".join(f"{item['sha256']}  {item['path']}\n" for item in checkpoints))
    atomic_json(CAMPAIGN/"selection/TEST_UNLOCK.json",{"project":"ARGOS v2","authorized":True,"frozen_recipe_sha256":sha256(path),"run_integrity_sha256":sha256(integrity_path),"selection_sha256":sha256(selection_path),"single_campaign_only":True,"dataset_7_opened":False})
if __name__=="__main__": main()
