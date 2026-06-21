#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import ROOT_DIR, EXTERNAL_DIR, DATASET_DIR, RESULTS_DIR

import cv2
import numpy as np
import torch


ROOT = ROOT_DIR
S2M2_REPO = EXTERNAL_DIR / "frame_stereo_repos/s2m2"
sys.path.insert(0, str(S2M2_REPO / "src"))

from s2m2.core.model.s2m2 import S2M2
from s2m2.core.utils.image_utils import image_crop, image_pad


def read_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


MODEL_CONFIGS = {
    "S": {"feature_channels": 128, "num_transformer": 1, "weight": "CH128NTR1.pth"},
    "M": {"feature_channels": 192, "num_transformer": 2, "weight": "CH192NTR2.pth"},
    "L": {"feature_channels": 256, "num_transformer": 3, "weight": "CH256NTR3.pth"},
    "XL": {"feature_channels": 384, "num_transformer": 3, "weight": "CH384NTR3.pth"},
}


def build_model(device, variant: str):
    cfg = MODEL_CONFIGS[variant]
    model = S2M2(feature_channels=cfg["feature_channels"], dim_expansion=1, num_transformer=cfg["num_transformer"], use_positivity=True, refine_iter=3)
    ckpt = torch.load(S2M2_REPO / "weights/pretrain_weights" / cfg["weight"], map_location="cpu", weights_only=True)
    model.my_load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def infer(model, left, right, device, width):
    orig_h, orig_w = left.shape[:2]
    scale_x = width / float(orig_w)
    new_h = int(round(orig_h * scale_x))
    left_in = cv2.resize(left, (width, new_h), interpolation=cv2.INTER_LINEAR)
    right_in = cv2.resize(right, (width, new_h), interpolation=cv2.INTER_LINEAR)
    left_t = torch.from_numpy(left_in).permute(2, 0, 1).unsqueeze(0).float().to(device)
    right_t = torch.from_numpy(right_in).permute(2, 0, 1).unsqueeze(0).float().to(device)
    h, w = left_t.shape[-2:]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        pred, _occ, _conf = model(image_pad(left_t, 32), image_pad(right_t, 32))
    if device.type == "cuda":
        torch.cuda.synchronize()
    pred = image_crop(pred, (h, w)).squeeze().float().cpu().numpy()
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale_x
    return np.clip(pred.astype(np.float32), 0, None), (time.perf_counter() - t0) * 1000.0, scale_x


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--sequences-root", type=Path, default=ROOT / "results/04_dataset_derivatives/SCARED/scared_long_sequences")
    p.add_argument("--out-root", type=Path, default=ROOT / "results/04_dataset_derivatives/SCARED/scared_long_predictions/s2m2_l736")
    p.add_argument("--variant", choices=sorted(MODEL_CONFIGS), default="L")
    p.add_argument("--width", type=int, default=736)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = build_model(device, args.variant)
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seq in sorted(d for d in args.sequences_root.iterdir() if d.is_dir() and not d.name.startswith("_")):
        left_paths = sorted((seq / "left").glob("*.png"))
        right_paths = sorted((seq / "right").glob("*.png"))
        if args.max_frames:
            left_paths = left_paths[: args.max_frames]
            right_paths = right_paths[: args.max_frames]
        out = args.out_root / seq.name
        disp_dir = out / "disp"
        disp_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for left_path, right_path in zip(left_paths, right_paths):
            dst = disp_dir / f"{left_path.stem}.npy"
            if dst.exists():
                pred = np.load(dst)
                runtime_ms = 0.0
                scale_x = args.width / float(pred.shape[1])
            else:
                pred, runtime_ms, scale_x = infer(model, read_rgb(left_path), read_rgb(right_path), device, args.width)
                np.save(dst, pred)
            rows.append({"frame_id": left_path.stem, "disp_path": str(dst), "runtime_ms": runtime_ms, "scale_x": scale_x, "shape": list(pred.shape)})
        meta = {
            "sequence_id": seq.name,
            "model": f"S2M2-{args.variant}@{args.width}",
            "variant": args.variant,
            "resize_width": args.width,
            "frames": len(rows),
            "coordinate_system": "original image disparity coordinates",
            "scale_y": rows[0]["scale_x"] if rows else None,
            "rows": rows,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        all_rows.append(meta)
    summary = {
        "device": str(device),
        "sequences": [{"sequence_id": m["sequence_id"], "frames": m["frames"]} for m in all_rows],
        "total_frames": sum(m["frames"] for m in all_rows),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0,
    }
    (args.out_root / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
