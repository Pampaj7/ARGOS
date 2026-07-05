#!/usr/bin/env python3
"""Build training-format refiner shards from an OOD sequence manifest.

Produces per-sequence .npz {raw_disp, gt_disp, valid_mask, delta_disp_gt_minus_raw}
at the SAME target grid (target_scale=0.25) and with the SAME valid-mask semantics as
the primary SCARED training targets, so every existing ARGOS refiner + its FullFrameDataset
consumes OOD data unchanged. No OOD-specific tuning: scale/units are physical conversions
only (identical recipe to in-domain).

Input:  a sequence_manifest.csv produced by an OOD adapter (servct_adapter.py / d4d_adapter.py)
Output: <prepared>/shards/<sequence_id>.npz  +  frame_targets_index.csv

Reuses generate_distillation_targets_selected_clips.{target_hw,valid_masked_downsample_disparity}.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/dtu/p1/leopam/ARGOS")
sys.path.insert(0, str(ROOT / "scripts/temporal_refinement"))
sys.path.insert(0, str(ROOT / "scripts/temporal_refinement/eval_scripts"))

from generate_distillation_targets_selected_clips import (  # noqa: E402
    target_hw,
    valid_masked_downsample_disparity,
)

TARGET_SCALE = 0.25      # identical to s2m2_gt_refiner_targets_full
MIN_VALID_RATIO = 0.25   # identical


def load_manifest(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def build(manifest_csv: Path, out_root: Path) -> None:
    rows = load_manifest(manifest_csv)
    shard_dir = out_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # group by sequence, preserve temporal order
    seqs: dict[str, list[dict]] = {}
    for r in rows:
        seqs.setdefault(r["sequence_id"], []).append(r)
    for sid in seqs:
        seqs[sid].sort(key=lambda r: int(r.get("order_index", r["frame_id"]) or 0))

    index_rows: list[dict] = []
    for sid, frames in seqs.items():
        raws, gts, valids, deltas = [], [], [], []
        for r in frames:
            raw_full = np.load(r["raw_disp_path"]).astype(np.float32)
            gt_full = np.load(r["gt_disp_path"]).astype(np.float32)
            vm_full = np.load(r["valid_mask_path"]).astype(bool)
            # training valid semantics (generate_s2m2_gt_refiner_targets_full L219/234)
            valid_full = vm_full & np.isfinite(gt_full) & (gt_full > 0) & np.isfinite(raw_full)
            out_h, out_w = target_hw(gt_full.shape, TARGET_SCALE)
            gt_ds, gvalid = valid_masked_downsample_disparity(gt_full, valid_full, out_h, out_w, MIN_VALID_RATIO)
            raw_ds, _ = valid_masked_downsample_disparity(raw_full, valid_full, out_h, out_w, MIN_VALID_RATIO)
            valid = gvalid & np.isfinite(gt_ds) & np.isfinite(raw_ds) & (gt_ds > 0)
            delta = np.where(valid, gt_ds - raw_ds, 0.0).astype(np.float32)
            raws.append(raw_ds.astype(np.float16))
            gts.append(gt_ds.astype(np.float16))
            valids.append(valid.astype(np.uint8))
            deltas.append(delta.astype(np.float16))

        shard_path = shard_dir / f"{sid}.npz"
        np.savez(
            shard_path,
            raw_disp=np.stack(raws), gt_disp=np.stack(gts),
            valid_mask=np.stack(valids), delta_disp_gt_minus_raw=np.stack(deltas),
        )
        for i, r in enumerate(frames):
            index_rows.append({
                "sequence_id": sid,
                "frame_id": r["frame_id"],
                "frame_index": i,
                "previous_frame_id": frames[i - 1]["frame_id"] if i > 0 else "",
                "next_frame_id": frames[i + 1]["frame_id"] if i < len(frames) - 1 else "",
                "target_path": str(shard_path.resolve()),
                "frame_offset": i,
                "target_h": out_h,
                "target_w": out_w,
                "dataset": r.get("dataset", ""),
                "split": r.get("split", ""),
                "continuity_flag": r.get("continuity_flag", ""),
            })

    idx_path = out_root / "frame_targets_index.csv"
    with idx_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)
    print(f"shards: {len(seqs)} sequences, {len(index_rows)} frames -> {shard_dir}")
    print(f"index: {idx_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "results/03_temporal_refinement/ood/prepared/servct/sequence_manifest.csv")
    p.add_argument("--out-root", type=Path,
                   default=ROOT / "results/03_temporal_refinement/ood/prepared/servct")
    args = p.parse_args()
    build(args.manifest, args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
