#!/usr/bin/env python3
import sys
import os
import json
import argparse
import subprocess
import glob
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.temporal_refinement.eval_scripts.evaluate_scared_temporal_gt import (
    read_metadata, method_metrics, write_csv, write_report
)

def safe_name(name: str) -> str:
    return name.replace("@", "_").replace("/", "_").replace(" ", "_")

def run_adapter(script_path: Path, seq_dir: Path, out_dir: Path, checkpoint: Path, name: str):
    pred_dir = out_dir / "predictions" / safe_name(name)
    pred_dir.mkdir(parents=True, exist_ok=True)
    meta_file = pred_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            m = json.load(f)
            m["method"] = name
            return m
            
    cmd = [
        "python3", str(script_path),
        "--sequence-dir", str(seq_dir),
        "--out-dir", str(pred_dir),
        "--checkpoint", str(checkpoint)
    ]
    print(f"Running video adapter {name}...")
    subprocess.run(cmd, check=True)
    with open(meta_file) as f:
        meta = json.load(f)
        meta["method"] = name
    return meta

def run_raft_like(repo_path: Path, script_name: str, seq_dir: Path, out_dir: Path, checkpoint: Path, name: str, extra_args: list = []):
    pred_dir = out_dir / "predictions" / safe_name(name)
    pred_dir.mkdir(parents=True, exist_ok=True)
    meta_file = pred_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            m = json.load(f)
            m["method"] = name
            return m

    # Convert glob pattern to list of files to avoid shell expansion issues, or just pass glob string if the script handles it using glob.glob.
    # RAFT, DEFOM, MonSter use glob.glob internally.
    cmd = [
        "python3", script_name,
        "--restore_ckpt", str(checkpoint),
        "--left_imgs", f"{seq_dir}/left/*.png",
        "--right_imgs", f"{seq_dir}/right/*.png",
        "--output_directory", str(pred_dir),
        "--save_numpy"
    ] + extra_args
    print(f"Running frame method {name}...", flush=True)
    # Add parent of repo_path to PYTHONPATH so models can be imported if they do relative imports
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_path) + ":" + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(repo_path), check=True, env=env)
    
    # Methods that save with .npy (RAFT-Stereo, DEFOM-Stereo, MonSter++) don't need renaming because the input files are 000000.png, so output is 000000.npy

    meta = {"method": name, "kind": "frame_stereo", "checkpoint": str(checkpoint)}
    with open(meta_file, "w") as f:
        json.dump(meta, f)
    return meta

def run_stereoanywhere(repo_path: Path, seq_dir: Path, out_dir: Path, checkpoint: Path, name: str):
    pred_dir = out_dir / "predictions" / safe_name(name)
    pred_dir.mkdir(parents=True, exist_ok=True)
    meta_file = pred_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            return json.load(f)

    left_imgs = sorted((seq_dir / "left").glob("*.png"))
    right_imgs = sorted((seq_dir / "right").glob("*.png"))
    
    print(f"Running frame method {name}...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_path) + ":" + env.get("PYTHONPATH", "")
    
    runtimes = []
    for l, r in zip(left_imgs, right_imgs):
        dst = pred_dir / f"{l.stem}_disparity.npy"
        if dst.exists(): continue
        
        cmd = [
            "python3", "demo.py",
            "--left", str(l),
            "--right", str(r),
            "--loadstereomodel", str(checkpoint),
            "--outdir", str(pred_dir)
        ]
        t0 = time.time()
        subprocess.run(cmd, cwd=str(repo_path), check=True, env=env)
        runtimes.append(time.time() - t0)
        
    meta = {
        "method": name,
        "avg_runtime_ms": (np.mean(runtimes) * 1000) if runtimes else np.nan,
        "causal": "No",
        "kind": "frame"
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f)
    return meta

def run_codd(repo_path: Path, seq_dir: Path, out_dir: Path, checkpoint: Path, name: str):
    pred_dir = out_dir / "predictions" / safe_name(name)
    pred_dir.mkdir(parents=True, exist_ok=True)
    meta_file = pred_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            return json.load(f)

    left_dir = seq_dir / "left"
    right_dir = seq_dir / "right"
    config = "configs/inference_config.py"

    cmd = [
        "conda", "run", "-n", "codd", "--no-capture-output",
        "python3", "inference.py",
        config,
        str(checkpoint),
        "--img-dir", str(left_dir),
        "--r-img-dir", str(right_dir),
        "--show-dir", str(pred_dir),
        "--num-frames", "-1"
    ]
    print(f"Running video method {name}...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_path) + ":" + env.get("PYTHONPATH", "")
    
    t0 = time.time()
    subprocess.run(cmd, cwd=str(repo_path), check=True, env=env)
    runtime = time.time() - t0

    num_frames = len(list(left_dir.glob("*.png")))
    meta = {
        "method": name,
        "avg_runtime_ms": (runtime / max(1, num_frames)) * 1000,
        "causal": "No (bidirectional)",
        "kind": "video"
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f)
    return meta

def load_predictions(out_dir: Path, method: str, frames: list[dict]) -> list[np.ndarray]:
    pred_dir = out_dir / "predictions" / safe_name(method)
    preds = []
    for frame in frames:
        fid = frame['id']
        path = pred_dir / f"{fid}.npy"
        if not path.exists():
            path_png = pred_dir / f"{fid}.png.npy"
            if path_png.exists(): path = path_png
        if not path.exists():
            path_disp = pred_dir / f"{fid}_disparity.npy"
            if path_disp.exists(): path = path_disp
        if not path.exists():
            path_int = pred_dir / f"{int(fid)}.npy"
            if path_int.exists(): path = path_int
        if not path.exists():
            # For DEFOM
            path_defom = list(pred_dir.glob(f"{fid}.png_*.npy"))
            if path_defom: path = path_defom[0]
        if not path.exists():
            # For CODD
            path_codd = pred_dir / f"{fid}.disp.pred.npz"
            if path_codd.exists():
                preds.append(np.load(path_codd)["disp"].astype(np.float32))
                continue
        pred = np.load(path).astype(np.float32)
        if method == "MonSter++":
            pred = pred / 256.0
        elif method == "RAFT-Stereo":
            pred = np.abs(pred)
        preds.append(pred)
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, default=ROOT / "dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/02_video_stereo/all_methods_fair_eval")
    parser.add_argument("--min-valid-ratio", type=float, default=0.2)
    args = parser.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = read_metadata(args.sequence_root, 0)
    
    method_meta = []
    
    # 1. TCSM
    tcsm_ckpt = ROOT / "external/video_stereo_repos/Temporally-Consistent-Stereo-Matching/checkpoints/sceneflow.pth"
    if tcsm_ckpt.exists():
        try:
            method_meta.append(run_adapter(ROOT / "scripts/temporal_refinement/adapters/run_tcsm_temporal.py", args.sequence_root, args.out_dir, tcsm_ckpt, "TCSM"))
        except Exception as e:
            print(f"Failed TCSM: {e}")

    # 2. PPMStereo
    ppm_ckpt = ROOT / "external/video_stereo_repos/PPMStereo/ckpt/ppmstereo_weights/ppmstereo_sf.pth"
    if ppm_ckpt.exists():
        try:
            method_meta.append(run_adapter(ROOT / "scripts/temporal_refinement/adapters/run_ppmstereo_temporal.py", args.sequence_root, args.out_dir, ppm_ckpt, "PPMStereo"))
        except Exception as e:
            print(f"Failed PPMStereo: {e}")

    # 3. RAFT-Stereo
    raft_ckpt = ROOT / "external/frame_stereo_repos/RAFT-Stereo/models/raftstereo-sceneflow.pth"
    if raft_ckpt.exists():
        try:
            method_meta.append(run_raft_like(ROOT / "external/frame_stereo_repos/RAFT-Stereo", "demo.py", args.sequence_root, args.out_dir, raft_ckpt, "RAFT-Stereo", ["--mixed_precision"]))
        except Exception as e:
            print(f"Failed RAFT-Stereo: {e}")

    # 4. DEFOM-Stereo
    defom_ckpt = ROOT / "external/frame_stereo_repos/DEFOM-Stereo/checkpoints/defomstereo_vitl_eth3d.pth"
    if defom_ckpt.exists():
        try:
            method_meta.append(run_raft_like(ROOT / "external/frame_stereo_repos/DEFOM-Stereo", "demo.py", args.sequence_root, args.out_dir, defom_ckpt, "DEFOM-Stereo", []))
        except Exception as e:
            print(f"Failed DEFOM-Stereo: {e}")

    # 5. MonSter++
    monster_ckpt = ROOT / "external/frame_stereo_repos/MonSter-plusplus/MonSter++/checkpoints/Mix_all_large.pth"
    if monster_ckpt.exists():
        try:
            method_meta.append(run_raft_like(ROOT / "external/frame_stereo_repos/MonSter-plusplus/MonSter++", "save_disp.py", args.sequence_root, args.out_dir, monster_ckpt, "MonSter++", []))
        except Exception as e:
            print(f"Failed MonSter++: {e}")

    # 6. StereoAnywhere
    stereoanywhere_ckpt = ROOT / "external/frame_stereo_repos/stereoanywhere/weights/stereoanywhere_sceneflow.pth"
    if stereoanywhere_ckpt.exists():
        try:
            method_meta.append(run_stereoanywhere(ROOT / "external/frame_stereo_repos/stereoanywhere", args.sequence_root, args.out_dir, stereoanywhere_ckpt, "StereoAnywhere"))
        except Exception as e:
            print(f"Failed StereoAnywhere: {e}")

    # 7. CODD
    codd_ckpt = ROOT / "external/video_stereo_repos/codd/checkpoints/codd_flyingthings3d.pth"
    if codd_ckpt.exists():
        try:
            method_meta.append(run_codd(ROOT / "external/video_stereo_repos/codd", args.sequence_root, args.out_dir, codd_ckpt, "CODD"))
        except Exception as e:
            print(f"Failed CODD: {e}")

    # TODO: Add S2M2 and SAV directly here, or copy their predictions from the previous eval folder to ensure they are on the exact same table.
    
    # We can also add S2M2 by importing it if we are in the right python env
    try:
        sys.path.insert(0, str(ROOT / "scripts/temporal_refinement/eval_scripts"))
        from evaluate_scared_temporal_gt import run_s2m2, run_sav, build_s2m2
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        method_meta.append(run_s2m2(frames, args.out_dir, "S", 512, __import__("torch").device(device)))
        method_meta.append(run_s2m2(frames, args.out_dir, "L", 736, __import__("torch").device(device)))
        method_meta.append(run_sav(frames, args.out_dir, __import__("torch").device(device), 32, 4))
    except Exception as e:
        print(f"Failed to run S2M2/SAV directly: {e}")
    
    summaries = []
    per_frame_all = []
    methods = [m["method"] for m in method_meta]
    
    for meta in method_meta:
        try:
            preds = load_predictions(args.out_dir, meta["method"], frames)
            summary, per_frame = method_metrics(meta["method"], preds, frames, args.min_valid_ratio, meta)
            summaries.append(summary)
            per_frame_all.extend(per_frame)
        except Exception as e:
            print(f"Failed to compute metrics for {meta['method']}: {e}", flush=True)
        
    try:
        write_csv(args.out_dir / "temporal_evaluation_gt.csv", summaries)
        write_csv(args.out_dir / "per_frame_metrics.csv", per_frame_all)
        write_report(args.out_dir, summaries, args.sequence_root, methods)
        print("Done! Report written to:", args.out_dir / "temporal_evaluation_gt.md", flush=True)
    except Exception as e:
        print(f"Failed to write report: {e}", flush=True)

if __name__ == "__main__":
    main()
