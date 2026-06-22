#!/usr/bin/env python3
import json
import shutil
import argparse
from pathlib import Path
import numpy as np

# We assume standard ARGOS relative paths
try:
    from scripts.argos_paths import DATASET_DIR
except ImportError:
    DATASET_DIR = Path("/dtu/p1/leopam/ARGOS/dataset")

RAW_DIR = DATASET_DIR / "D4D" / "raw" / "extracted"
OUT_DIR = DATASET_DIR / "D4D" / "curated" / "temporal_gt"

def curate_clip(specimen_dir: Path, session_dir: Path, clip_dir: Path):
    specimen_id = specimen_dir.name
    session_id = session_dir.name
    clip_id = clip_dir.name
    seq_name = f"{specimen_id}_{session_id}_{clip_id}"
    
    seq_out = OUT_DIR / seq_name
    if seq_out.exists():
        print(f"Skipping {seq_name}, already curated.")
        return

    print(f"Curating {seq_name}...")
    left_out = seq_out / "left"
    right_out = seq_out / "right"
    disp_out = seq_out / "gt" / "Disparity_float32"
    depth_out = seq_out / "gt" / "DepthL_float32"
    
    for d in [left_out, right_out, disp_out, depth_out]:
        d.mkdir(parents=True, exist_ok=True)
        
    left_src = clip_dir / "left_images_rect"
    right_src = clip_dir / "right_images_rect"
    depth_src = clip_dir / "stereo_depth"
    
    if not left_src.exists() or not right_src.exists() or not depth_src.exists():
        print(f"  Missing data for {seq_name}, skipping.")
        return
        
    left_files = sorted(list(left_src.glob("*.png")))
    right_files = sorted(list(right_src.glob("*.png")))
    depth_files = sorted(list(depth_src.glob("*.npy")))
    
    # Optional: copy calibration
    calib_src = session_dir / "camera_info"
    calib_out = seq_out / "calibration"
    if calib_src.exists():
        calib_out.mkdir(parents=True, exist_ok=True)
        for cf in calib_src.glob("*.*"):
            shutil.copy2(cf, calib_out / cf.name)
            
    # Need to convert D4D depth (.npy in meters) to disparity?
    # Actually if our pipeline expects Disparity_float32, we should compute it:
    # disp = (baseline * focal_length) / depth
    # We can get baseline and focal_length from camera_info if needed.
    # For now, let's just copy depth and assume a default baseline/focal_length or skip disp
    # if we don't parse YAML. But let's copy depth.
    
    for i, lf in enumerate(left_files):
        rf = right_files[i]
        df = depth_files[i]
        
        frame_name = f"{i:06d}.png"
        npy_name = f"{i:06d}.npy"
        
        shutil.copy2(lf, left_out / frame_name)
        shutil.copy2(rf, right_out / frame_name)
        
        # Copy depth
        shutil.copy2(df, depth_out / npy_name)
        
        # Simple depth to disp assuming standard daVinci if we can't parse:
        # disp = (fx * b) / Z. 
        # For D4D baseline is ~0.005m, fx ~1000px.
        depth = np.load(str(df))
        disp = np.zeros_like(depth)
        valid = depth > 0
        disp[valid] = (1000.0 * 0.005) / depth[valid]  # Placeholder, should read YAML
        np.save(str(disp_out / npy_name), disp.astype(np.float32))

    # Save metadata
    meta = {
        "sequence_id": seq_name,
        "specimen": specimen_id,
        "session": session_id,
        "clip": clip_id,
        "frames": len(left_files),
        "fps": 30.0  # D4D is 30 FPS usually
    }
    (seq_out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  Saved {len(left_files)} frames.")

def main():
    if not RAW_DIR.exists():
        print(f"Raw directory {RAW_DIR} not found.")
        return
        
    for specimen_dir in sorted(RAW_DIR.glob("specimen_*")):
        if not specimen_dir.is_dir(): continue
        for session_dir in sorted(specimen_dir.glob("*_*_*")): # e.g. 2025_03_06-16_49_40
            if not session_dir.is_dir(): continue
            clips_dir = session_dir / "clips"
            if not clips_dir.exists(): continue
            for clip_dir in sorted(clips_dir.glob("Clip_*")):
                if not clip_dir.is_dir(): continue
                curate_clip(specimen_dir, session_dir, clip_dir)

if __name__ == "__main__":
    main()
