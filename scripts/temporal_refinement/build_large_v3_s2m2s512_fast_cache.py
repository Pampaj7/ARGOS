#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import random
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import ROOT_DIR, EXTERNAL_DIR, DATASET_DIR, RESULTS_DIR

import cv2
import numpy as np

from scripts.temporal_refinement.build_debug_cache import colorize


ROOT = ROOT_DIR
SEQ_ROOT = ROOT / "results/04_dataset_derivatives/SCARED/scared_long_sequences"
S_ROOT = ROOT / "results/04_dataset_derivatives/SCARED/scared_long_predictions/s2m2_s512"
L_ROOT = ROOT / "results/04_dataset_derivatives/SCARED/scared_long_predictions/s2m2_l736"
SAV_ROOT = ROOT / "results/04_dataset_derivatives/SCARED/scared_long_predictions/stereoanyvideo_384x640"
OUT = ROOT / "results/03_temporal_refinement/cache/large_v3_s2m2s512_fast"


def rel(path: Path) -> str:
    return str(path.relative_to(OUT))


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def save_f16(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    arr = np.load(src).astype(np.float16)
    np.save(dst, arr)


def add_stats(prefix: str, arr: np.ndarray, out: dict):
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    out[f"{prefix}_min"] = float(np.nanmin(arr[finite])) if finite.any() else None
    out[f"{prefix}_max"] = float(np.nanmax(arr[finite])) if finite.any() else None
    out[f"{prefix}_mean"] = float(np.nanmean(arr[finite])) if finite.any() else None


def build_sequence(seq: Path) -> tuple[list[dict], dict]:
    seq_out = OUT / seq.name
    rgb_dir = seq_out / "left_rgb"
    s_dir = seq_out / "s2m2_s512_disp"
    l_dir = seq_out / "s2m2_l736_disp"
    sav_dir = seq_out / "sav_disp"
    frame_ids = [p.stem for p in sorted((seq / "left").glob("*.png"))]
    for fid in frame_ids:
        link_or_copy(seq / "left" / f"{fid}.png", rgb_dir / f"{fid}.png")
        save_f16(S_ROOT / seq.name / "disp" / f"{fid}.npy", s_dir / f"{fid}.npy")
        save_f16(L_ROOT / seq.name / "disp" / f"{fid}.npy", l_dir / f"{fid}.npy")
        save_f16(SAV_ROOT / seq.name / "disp" / f"{fid}.npy", sav_dir / f"{fid}.npy")
    rows = []
    diffs_sl = []
    diffs_ssav = []
    for center_idx in range(2, len(frame_ids) - 2):
        ids = frame_ids[center_idx - 2 : center_idx + 3]
        center = ids[2]
        s = np.load(s_dir / f"{center}.npy").astype(np.float32)
        l = np.load(l_dir / f"{center}.npy").astype(np.float32)
        sav = np.load(sav_dir / f"{center}.npy").astype(np.float32)
        diff_sl = float(np.nanmean(np.abs(s - l)))
        diff_ssav = float(np.nanmean(np.abs(s - sav)))
        diffs_sl.append(diff_sl)
        diffs_ssav.append(diff_ssav)
        row = {
            "sample_id": "",
            "sequence_id": seq.name,
            "center_frame_id": center,
            "frame_tminus2": ids[0],
            "frame_tminus1": ids[1],
            "frame_t": ids[2],
            "frame_tplus1": ids[3],
            "frame_tplus2": ids[4],
            "rgb_center_path": rel(rgb_dir / f"{center}.png"),
            "s2m2_s512_tminus2_path": rel(s_dir / f"{ids[0]}.npy"),
            "s2m2_s512_tminus1_path": rel(s_dir / f"{ids[1]}.npy"),
            "s2m2_s512_t_path": rel(s_dir / f"{ids[2]}.npy"),
            "s2m2_s512_tplus1_path": rel(s_dir / f"{ids[3]}.npy"),
            "s2m2_s512_tplus2_path": rel(s_dir / f"{ids[4]}.npy"),
            "s2m2_l736_t_path": rel(l_dir / f"{center}.npy"),
            "sav_tminus1_path": rel(sav_dir / f"{ids[1]}.npy"),
            "sav_t_path": rel(sav_dir / f"{ids[2]}.npy"),
            "sav_tplus1_path": rel(sav_dir / f"{ids[3]}.npy"),
            "height": s.shape[0],
            "width": s.shape[1],
            "has_gt": False,
            "mean_abs_s2m2s_s2m2l_diff": diff_sl,
            "mean_abs_s2m2s_sav_diff": diff_ssav,
        }
        add_stats("s2m2s_center_disp", s, row)
        add_stats("s2m2l_center_disp", l, row)
        add_stats("sav_center_disp", sav, row)
        rows.append(row)
    meta = {
        "sequence_id": seq.name,
        "frames": len(frame_ids),
        "valid_centers": len(rows),
        "mean_abs_s2m2s_s2m2l_diff": float(np.mean(diffs_sl)) if diffs_sl else None,
        "mean_abs_s2m2s_sav_diff": float(np.mean(diffs_ssav)) if diffs_ssav else None,
    }
    (seq_out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return rows, meta


def montage(row: dict):
    rgb = cv2.imread(str(OUT / row["rgb_center_path"]), cv2.IMREAD_COLOR)
    s = np.load(OUT / row["s2m2_s512_t_path"]).astype(np.float32)
    l = np.load(OUT / row["s2m2_l736_t_path"]).astype(np.float32)
    sav = np.load(OUT / row["sav_t_path"]).astype(np.float32)
    vmax = float(np.nanpercentile(np.concatenate([s.ravel(), l.ravel(), sav.ravel()]), 99))
    tiles = [
        rgb,
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
    (OUT / "sanity_montages").mkdir(exist_ok=True)
    rows = []
    seq_metas = []
    for seq in sorted(d for d in SEQ_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")):
        seq_rows, seq_meta = build_sequence(seq)
        seq_metas.append(seq_meta)
        rows.extend(seq_rows)
    for sample_id, row in enumerate(rows):
        row["sample_id"] = sample_id
    with (OUT / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    random.seed(17)
    for row in random.sample(rows, min(20, len(rows))):
        montage(row)
    sl = np.array([float(r["mean_abs_s2m2s_s2m2l_diff"]) for r in rows], dtype=np.float32)
    ssav = np.array([float(r["mean_abs_s2m2s_sav_diff"]) for r in rows], dtype=np.float32)
    meta = {
        "cache_name": "large_v3_s2m2s512_fast",
        "format": "indexed_per_frame_float16_npy",
        "sample_count": len(rows),
        "sequence_count": len(seq_metas),
        "window_size": 5,
        "backbone": "S2M2-S@512",
        "spatial_teacher": "S2M2-L@736",
        "temporal_teacher": "StereoAnyVideo@384x640",
        "coordinate_system": "original image disparity coordinates",
        "sequences": seq_metas,
        "quick_statistics": {
            "mean_abs_s2m2s_s2m2l_diff_mean": float(sl.mean()),
            "mean_abs_s2m2s_sav_diff_mean": float(ssav.mean()),
            "mean_abs_s2m2s_s2m2l_diff_min": float(sl.min()),
            "mean_abs_s2m2s_sav_diff_min": float(ssav.min()),
            "mean_abs_s2m2s_s2m2l_diff_max": float(sl.max()),
            "mean_abs_s2m2s_sav_diff_max": float(ssav.max()),
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Large V3 S2M2-S@512 Fast Cache\n\n"
        "Indexed per-frame cache for multi-teacher temporal refinement.\n\n"
        f"- Samples: `{len(rows)}`\n"
        f"- Sequences: `{len(seq_metas)}`\n"
        "- Disparity storage: float16 `.npy` per frame\n"
        "- RGB storage: symlink/copy to extracted PNG frames\n"
        "- Backbone: `S2M2-S@512`\n"
        "- Spatial teacher: `S2M2-L@736`\n"
        "- Temporal teacher: `StereoAnyVideo@384x640`\n\n"
        "Payload arrays and RGB frame links are ignored by Git.\n"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
