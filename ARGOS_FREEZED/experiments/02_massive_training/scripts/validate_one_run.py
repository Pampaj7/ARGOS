#!/usr/bin/env python3
"""D2-only validation entry point; campaign aggregator performs shared-bank evaluation."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from campaign_common import *
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--budget",type=int,choices=(1,3,6),required=True); parser.add_argument("--seed",type=int,choices=SEEDS,required=True); args=parser.parse_args()
    verify_frozen_core(); guard_no_d7(VALIDATION_SEQUENCES)
    checkpoint=run_directory(args.budget,args.seed)/"checkpoints/best_validation.pt"
    if not checkpoint.is_file(): raise SystemExit(f"missing checkpoint: {checkpoint}")
    raise SystemExit("Use aggregate_campaign.py --validate-d2 to evaluate all checkpoints against one shared D2 bank.")
if __name__=="__main__": main()
