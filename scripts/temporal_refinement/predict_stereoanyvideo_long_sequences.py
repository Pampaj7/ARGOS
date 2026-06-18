#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import ROOT_DIR, EXTERNAL_DIR, DATASET_DIR, RESULTS_DIR

import cv2
import numpy as np


ROOT = ROOT_DIR
SAV_SCRIPT = Path("../../external/video_stereo_repos/stereoanyvideo/scripts/run_argos_scared_smoke.py")
SAV_CKPT = Path("../../external/video_stereo_repos/stereoanyvideo/checkpoints/StereoAnyVideo_MIX.pth")


def run_chunk(seq: Path, frame_ids: list[str], temp: Path, out_chunk: Path):
    if temp.exists():
        shutil.rmtree(temp)
    (temp / "left").mkdir(parents=True)
    (temp / "right").mkdir(parents=True)
    for idx, fid in enumerate(frame_ids):
        shutil.copy2(seq / "left" / f"{fid}.png", temp / "left" / f"{idx:06d}.png")
        shutil.copy2(seq / "right" / f"{fid}.png", temp / "right" / f"{idx:06d}.png")
    cmd = [
        "../../external/frame_stereo_repos/.miniconda/envs/argos/bin/python",
        str(SAV_SCRIPT),
        "--input",
        str(temp.resolve()),
        "--checkpoint",
        str(SAV_CKPT),
        "--output",
        str(out_chunk.resolve()),
        "--resize",
        "384x640",
        "--iters",
        "6",
    ]
    t0 = time.perf_counter()
    subprocess.check_call(cmd)
    return time.perf_counter() - t0


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--sequences-root", type=Path, default=ROOT / "results/04_dataset_derivatives/SCARED/scared_long_sequences")
    p.add_argument("--out-root", type=Path, default=ROOT / "results/04_dataset_derivatives/SCARED/scared_long_predictions/stereoanyvideo_384x640")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    temp_root = args.out_root / "_tmp_chunks"
    summaries = []
    for seq in sorted(d for d in args.sequences_root.iterdir() if d.is_dir() and not d.name.startswith("_")):
        left_paths = sorted((seq / "left").glob("*.png"))
        if args.max_frames:
            left_paths = left_paths[: args.max_frames]
        frame_ids = [p.stem for p in left_paths]
        out = args.out_root / seq.name
        disp_dir = out / "disp"
        disp_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        cursor = 0
        chunk_idx = 0
        while cursor < len(frame_ids):
            chunk_ids = frame_ids[cursor : min(cursor + args.chunk_size, len(frame_ids))]
            chunk_out = out / "chunks" / f"chunk_{chunk_idx:04d}"
            disparity_path = chunk_out / "disparity.npy"
            if not disparity_path.exists():
                runtime = run_chunk(seq, chunk_ids, temp_root / seq.name, chunk_out)
            else:
                runtime = 0.0
            disp = np.load(disparity_path).astype(np.float32)
            # SAV output is at 384x640. Rescale to original 1024x1280 and original disparity coordinates.
            for local_idx, fid in enumerate(chunk_ids):
                if (disp_dir / f"{fid}.npy").exists():
                    continue
                pred = disp[local_idx, 0] if disp.ndim == 4 else disp[local_idx]
                left = cv2.imread(str(seq / "left" / f"{fid}.png"), cv2.IMREAD_COLOR)
                h, w = left.shape[:2]
                scale_x = pred.shape[1] / float(w)
                pred_orig = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR) / scale_x
                np.save(disp_dir / f"{fid}.npy", pred_orig.astype(np.float32))
            rows.append({"chunk": chunk_idx, "frames": len(chunk_ids), "runtime_seconds": runtime})
            if cursor + args.chunk_size >= len(frame_ids):
                break
            cursor += args.chunk_size - args.overlap
            chunk_idx += 1
        frame_rows = [{"frame_id": fid, "disp_path": str(disp_dir / f"{fid}.npy")} for fid in frame_ids if (disp_dir / f"{fid}.npy").exists()]
        meta = {
            "sequence_id": seq.name,
            "model": "StereoAnyVideo@384x640",
            "frames": len(frame_rows),
            "coordinate_system": "original image disparity coordinates",
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "chunks": rows,
            "rows": frame_rows,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        summaries.append({"sequence_id": seq.name, "frames": len(frame_rows)})
    if temp_root.exists():
        shutil.rmtree(temp_root)
    summary = {"sequences": summaries, "total_frames": sum(s["frames"] for s in summaries)}
    (args.out_root / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
