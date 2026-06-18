#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

from scripts.temporal_refinement.build_debug_cache import colorize


ROOT = Path("/dtu/p1/leopam/ARGOS")
SEQ_ROOT = ROOT / "results/04_dataset_derivatives/SCARED/scared_long_sequences"
S_ROOT = ROOT / "results/04_dataset_derivatives/SCARED/scared_long_predictions/s2m2_s512"
L_ROOT = ROOT / "results/04_dataset_derivatives/SCARED/scared_long_predictions/s2m2_l736"
SAV_ROOT = ROOT / "results/04_dataset_derivatives/SCARED/scared_long_predictions/stereoanyvideo_384x640"
OUT = ROOT / "results/03_temporal_refinement/cache/large_v2_s2m2s512"


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def add_stats(prefix: str, x: np.ndarray, row: dict):
    finite = np.isfinite(x)
    if not finite.any():
        row[f"{prefix}_min"] = float("nan")
        row[f"{prefix}_max"] = float("nan")
        row[f"{prefix}_mean"] = float("nan")
        return
    row[f"{prefix}_min"] = float(np.min(x[finite]))
    row[f"{prefix}_max"] = float(np.max(x[finite]))
    row[f"{prefix}_mean"] = float(np.mean(x[finite]))


def load_disp(root: Path, seq: str, fid: str) -> np.ndarray:
    return np.load(root / seq / "disp" / f"{fid}.npy").astype(np.float32)


def write_sample(sample_id: int, seq: Path, center_idx: int, frame_ids: list[str]) -> dict:
    center = frame_ids[center_idx]
    window_ids = frame_ids[center_idx - 2 : center_idx + 3]
    rgb = read_rgb(seq / "left" / f"{center}.png")
    h, w = rgb.shape[:2]
    s_window = np.stack([load_disp(S_ROOT, seq.name, fid) for fid in window_ids])
    sav_window = np.stack([load_disp(SAV_ROOT, seq.name, fid) for fid in window_ids])
    l_center = load_disp(L_ROOT, seq.name, center)
    if s_window.shape != (5, h, w) or sav_window.shape != (5, h, w) or l_center.shape != (h, w):
        raise RuntimeError(f"Shape mismatch {seq.name} {center}: S={s_window.shape} L={l_center.shape} SAV={sav_window.shape} image={(h,w)}")
    sample_name = f"sample_{sample_id:06d}.npz"
    np.savez_compressed(
        OUT / "samples" / sample_name,
        center_rgb=rgb.astype(np.uint8),
        s2m2_s512_disp_window=s_window,
        s2m2_s512_disp_center=s_window[2],
        s2m2_l736_disp_center=l_center,
        stereoanyvideo_disp_center=sav_window[2],
        stereoanyvideo_disp_window=sav_window,
        stereoanyvideo_disp_prev=sav_window[1],
        stereoanyvideo_disp_cur=sav_window[2],
        frame_ids=np.array(window_ids),
        center_frame_id=np.array(center),
        source_sequence=np.array(seq.name),
        has_gt=np.array(False),
        scale_x=np.array(1.0, dtype=np.float32),
        scale_y=np.array(1.0, dtype=np.float32),
    )
    s_center = s_window[2]
    sav_center = sav_window[2]
    diff_sl = np.abs(s_center - l_center)
    diff_ssav = np.abs(s_center - sav_center)
    row = {
        "sample_id": sample_id,
        "sample_path": f"samples/{sample_name}",
        "source_sequence": seq.name,
        "center_index": center_idx,
        "center_frame_id": center,
        "window_frame_ids": " ".join(window_ids),
        "height": h,
        "width": w,
        "has_gt": False,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "mean_abs_s2m2s_s2m2l_diff": float(np.nanmean(diff_sl)),
        "mean_abs_s2m2s_sav_diff": float(np.nanmean(diff_ssav)),
    }
    add_stats("s2m2s_center_disp", s_center, row)
    add_stats("s2m2l_center_disp", l_center, row)
    add_stats("sav_center_disp", sav_center, row)
    return row


def montage(row: dict):
    sample = np.load(OUT / row["sample_path"])
    rgb = sample["center_rgb"]
    s = sample["s2m2_s512_disp_center"]
    l = sample["s2m2_l736_disp_center"]
    sav = sample["stereoanyvideo_disp_center"]
    vmax = float(np.nanpercentile(np.concatenate([s.ravel(), l.ravel(), sav.ravel()]), 99))
    tiles = [
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        colorize(s, vmax),
        colorize(l, vmax),
        colorize(sav, vmax),
        colorize(np.abs(s - l), 8.0, cv2.COLORMAP_MAGMA),
        colorize(np.abs(s - sav), 8.0, cv2.COLORMAP_MAGMA),
    ]
    labels = ["RGB", "S2M2-S@512", "S2M2-L@736", "StereoAnyVideo", "|S-L|", "|S-SAV|"]
    small = []
    for tile, label in zip(tiles, labels):
        tile = cv2.resize(tile, (190, 152), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        small.append(tile)
    cv2.imwrite(str(OUT / "sanity_montages" / f"sample_{int(row['sample_id']):06d}.png"), np.concatenate(small, axis=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    (OUT / "sanity_montages").mkdir(exist_ok=True)
    rows = []
    sample_id = 0
    for seq in sorted(d for d in SEQ_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")):
        left_ids = [p.stem for p in sorted((seq / "left").glob("*.png"))]
        ids = [
            fid
            for fid in left_ids
            if (S_ROOT / seq.name / "disp" / f"{fid}.npy").exists()
            and (L_ROOT / seq.name / "disp" / f"{fid}.npy").exists()
            and (SAV_ROOT / seq.name / "disp" / f"{fid}.npy").exists()
        ]
        for center_idx in range(2, len(ids) - 2):
            rows.append(write_sample(sample_id, seq, center_idx, ids))
            sample_id += 1
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (OUT / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    random.seed(11)
    for row in random.sample(rows, min(20, len(rows))):
        montage(row)
    by_seq = {}
    for row in rows:
        by_seq[row["source_sequence"]] = by_seq.get(row["source_sequence"], 0) + 1
    sl = np.array([r["mean_abs_s2m2s_s2m2l_diff"] for r in rows], dtype=np.float32)
    ssav = np.array([r["mean_abs_s2m2s_sav_diff"] for r in rows], dtype=np.float32)
    meta = {
        "cache_name": "large_v2_s2m2s512",
        "sample_count": len(rows),
        "samples_by_sequence": by_seq,
        "coordinate_system": "original image disparity coordinates",
        "window_size": 5,
        "backbone": "S2M2-S@512",
        "spatial_teacher": "S2M2-L@736",
        "temporal_teacher": "StereoAnyVideo@384x640",
        "quick_statistics": {
            "mean_abs_s2m2s_s2m2l_diff_mean": float(sl.mean()) if len(sl) else None,
            "mean_abs_s2m2s_s2m2l_diff_min": float(sl.min()) if len(sl) else None,
            "mean_abs_s2m2s_s2m2l_diff_max": float(sl.max()) if len(sl) else None,
            "mean_abs_s2m2s_sav_diff_mean": float(ssav.mean()) if len(ssav) else None,
            "mean_abs_s2m2s_sav_diff_min": float(ssav.min()) if len(ssav) else None,
            "mean_abs_s2m2s_sav_diff_max": float(ssav.max()) if len(ssav) else None,
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Temporal Refinement Cache Large V2 S2M2-S@512\n\n"
        f"Samples: `{len(rows)}`.\n\n"
        "Backbone: `S2M2-S@512`.\n"
        "Spatial teacher: `S2M2-L@736`.\n"
        "Temporal teacher: `StereoAnyVideo@384x640`.\n\n"
        "Payload `.npz` files are ignored by Git.\n"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
