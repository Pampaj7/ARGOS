#!/usr/bin/env python3
"""Fail-closed one-shot D7 plan; execution requires a validated evaluator."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from campaign_common import *
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--plan",action="store_true"); parser.add_argument("--execute",action="store_true"); args=parser.parse_args()
    if args.plan == args.execute: raise SystemExit("choose exactly one of --plan or --execute")
    verify_frozen_core(); recipe=CAMPAIGN/"selection/frozen_recipe_manifest.json"; unlock=CAMPAIGN/"selection/TEST_UNLOCK.json"; integrity=CAMPAIGN/"aggregate/run_integrity.json"; selection=CAMPAIGN/"selection/validation_selection_results.json"
    if not all(path.is_file() for path in (recipe,unlock,integrity,selection)): raise SystemExit("D7 LOCKED: recipe/unlock/integrity/selection missing")
    if not json.loads(integrity.read_text()).get("integrity_pass") or json.loads(selection.read_text()).get("verdict")!="ELIGIBLE BUDGET SELECTED": raise SystemExit("D7 LOCKED: D2 eligibility/integrity failed")
    authorization=json.loads(unlock.read_text())
    if not authorization.get("authorized") or authorization.get("frozen_recipe_sha256")!=sha256(recipe) or authorization.get("run_integrity_sha256")!=sha256(integrity): raise SystemExit("D7 LOCKED: invalid unlock")
    if (CAMPAIGN/"frozen_test/test_opened.json").exists(): raise SystemExit("D7 campaign already opened; refusing second execution")
    plan={"project":"ARGOS v2","mode":"D7 one-shot","selected_checkpoints":json.loads(recipe.read_text())["selected_checkpoints"],"methods":["raw","H4","canonical_geometry_v1","selected_seed_checkpoints"],"backbones":["S2M2-S","RAFT-Stereo","StereoAnywhere","CREStereo","Fast-FoundationStereo"],"artifacts":["frame_metrics.csv","per_backbone_metrics.csv","per_sequence_metrics.csv","summary.json","checkpoint_hashes.json"],"strict_common_support":"raw & H4 & canonical geometry_v1 & all selected checkpoints","dataset_7_opened":False}
    if args.plan: print(json.dumps(plan,sort_keys=True)); return
    raise SystemExit("D7 LOCKED: no extracted five-backbone evaluator has passed D2 parity; refusal preserves the one-shot test.")
if __name__=="__main__": main()
