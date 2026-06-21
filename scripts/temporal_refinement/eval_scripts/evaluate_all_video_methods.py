#!/usr/bin/env python3
import sys
import os
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.temporal_refinement.eval_scripts.evaluate_scared_temporal_gt import (
    read_metadata, method_metrics, write_csv, save_montages, write_report
)

def safe_name(name: str) -> str:
    return name.replace("@", "_").replace("/", "_").replace(" ", "_")

def run_adapter(script_path: Path, seq_dir: Path, out_dir: Path, checkpoint: Path, name: str):
    pred_dir = out_dir / "predictions" / safe_name(name)
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if predictions are already there
    meta_file = pred_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            return json.load(f)
            
    cmd = [
        "python3", str(script_path),
        "--sequence-dir", str(seq_dir),
        "--out-dir", str(pred_dir),
        "--checkpoint", str(checkpoint)
    ]
    print(f"Running {name}...")
    subprocess.run(cmd, check=True)
    
    # Read adapter's metadata.json
    with open(meta_file) as f:
        meta = json.load(f)
    return meta

def load_predictions(out_dir: Path, method: str, frames: list[dict]) -> list[np.ndarray]:
    pred_dir = out_dir / "predictions" / safe_name(method)
    return [np.load(pred_dir / f"{frame['id']}.npy").astype(np.float32) for frame in frames]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, default=ROOT / "dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/02_video_stereo/all_video_repos_fair_eval")
    parser.add_argument("--min-valid-ratio", type=float, default=0.2)
    args = parser.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = read_metadata(args.sequence_root, 0)
    
    # Prepare methods to run
    method_meta = []
    
    # 1. TCSM
    tcsm_ckpt = ROOT / "external/video_stereo_repos/Temporally-Consistent-Stereo-Matching/checkpoints/sceneflow.pth"
    if tcsm_ckpt.exists():
        meta = run_adapter(
            ROOT / "scripts/temporal_refinement/adapters/run_tcsm_temporal.py",
            args.sequence_root, args.out_dir, tcsm_ckpt, "TCSM"
        )
        method_meta.append(meta)
        
    # 2. PPMStereo
    ppm_ckpt = ROOT / "external/video_stereo_repos/PPMStereo/ckpt/ppmstereo_weights/ppmstereo_sf.pth"
    if ppm_ckpt.exists():
        meta = run_adapter(
            ROOT / "scripts/temporal_refinement/adapters/run_ppmstereo_temporal.py",
            args.sequence_root, args.out_dir, ppm_ckpt, "PPMStereo"
        )
        method_meta.append(meta)

    # Note: SAV is already done in fair_video_eval, we can just copy its predictions.
    # To keep it completely independent, we can let user see TCSM and PPMStereo first.
    
    summaries = []
    per_frame_all = []
    methods = [m["method"] for m in method_meta]
    
    for meta in method_meta:
        preds = load_predictions(args.out_dir, meta["method"], frames)
        summary, per_frame = method_metrics(meta["method"], preds, frames, args.min_valid_ratio, meta)
        summaries.append(summary)
        per_frame_all.extend(per_frame)
        
    write_csv(args.out_dir / "temporal_evaluation_gt.csv", summaries)
    write_csv(args.out_dir / "per_frame_metrics.csv", per_frame_all)
    
    write_report(args.out_dir, summaries, args.sequence_root, methods)
    print("Done! Report written to:", args.out_dir / "temporal_evaluation_gt.md")

if __name__ == "__main__":
    main()
