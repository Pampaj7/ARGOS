#!/usr/bin/env python3
"""Compact state-only monitor for ARGOS v2 budget runs."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from campaign_common import *
def main():
    rows=[]
    for budget in (1,3,6):
        for seed in SEEDS:
            path=run_directory(budget,seed)/"state.json"
            row=json.loads(path.read_text()) if path.is_file() else {"status":"pending","budget":f"{budget}x","seed":seed}
            rows.append(row)
    try: gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used","--format=csv,noheader"],text=True).strip().splitlines()
    except Exception: gpu=[]
    print(json.dumps({"project":"ARGOS v2","runs":rows,"gpus":gpu},indent=2))
if __name__=="__main__": main()
