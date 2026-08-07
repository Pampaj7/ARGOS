#!/usr/bin/env python3
"""Read-only ARGOS v2 cache/data audit; full banks remain ephemeral RAM."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from campaign_common import *
from data_pipeline import load_sequence_cache, load_sequence_info

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args()
    verify_frozen_core(); sequences=(TRAIN_SEQUENCES[:1]+VALIDATION_SEQUENCES[:1]) if args.smoke else TRAIN_SEQUENCES+VALIDATION_SEQUENCES
    guard_no_d7(sequences); rows=[]
    for sequence in sequences:
        info=load_sequence_info(sequence)
        for backbone in SEEN_BACKBONES:
            disparity,valid,ids,metadata=load_sequence_cache(backbone,sequence)
            checks={"sequence":sequence,"backbone":backbone,"shape":list(disparity.shape),"valid_shape":list(valid.shape),
                    "frame_order": [str(x) for x in ids]==info.frame_ids,"positive_left":metadata.get("disparity_convention")=="positive_left_disparity",
                    "finite_sample":bool(__import__('numpy').isfinite(disparity[[0,len(disparity)//2,-1]]).all())}
            checks["passed"]=all((checks["frame_order"],checks["positive_left"],checks["finite_sample"],tuple(disparity.shape[1:])==(144,180),valid.shape==disparity.shape)); rows.append(checks)
    if not all(row["passed"] for row in rows): raise RuntimeError("cache/data audit failed")
    atomic_json(CAMPAIGN/"protocol_audit/data_integrity.json",{"project":"ARGOS v2","dataset_7_opened":False,"rows":rows,"status":"PASS"})
    print(json.dumps({"status":"PASS","checks":len(rows)}))
if __name__=="__main__": main()
