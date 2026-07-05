#!/usr/bin/env python3
"""SERV-CT -> ARGOS OOD adapter (non-destructive).

Reshapes the already-ARGOS-format SERV-CT frames into the sequence layout that
`predict_s2m2_long_sequences.py` expects (seq/{left,right}/<frame>.png), so the
*unchanged* upstream S2M2-S generator can produce raw disparity in the exact same
recipe used for SCARED training (variant S, width 512, original image coords).

Emits a standardized sequence manifest with GT/calibration metadata. Uses symlinks;
never copies or mutates source data. GT convention (positive px, left reference,
rectified) already matches ARGOS, so no sign flip / rescale is applied.

Outputs under: results/03_temporal_refinement/ood/prepared/servct/
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/dtu/p1/leopam/ARGOS")
SRC = ROOT / "dataset/SERVCT/argos/servct_argos"
PREP = ROOT / "results/03_temporal_refinement/ood/prepared/servct"
# where the reused S2M2-S generator will (later) write raw disparity:
S2M2_OUT = PREP / "s2m2_s512"


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def build() -> None:
    PREP.mkdir(parents=True, exist_ok=True)
    seq_dir_root = PREP / "sequences"
    manifest: list[dict] = []
    seq_summ: dict[str, dict] = {}

    for split_dir in sorted(SRC.glob("*")):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for fr in sorted(split_dir.glob("*")):
            if not fr.is_dir():
                continue
            meta = json.loads((fr / "metadata.json").read_text())
            calib = json.loads((fr / "calib.json").read_text())
            exp = meta["sequence"]
            fid = meta["frame"]
            seq_id = f"{split}__{exp}"

            # sequence layout for the reused S2M2 generator
            link(fr / "left.png", seq_dir_root / seq_id / "left" / f"{fid}.png")
            link(fr / "right.png", seq_dir_root / seq_id / "right" / f"{fid}.png")

            raw_disp = S2M2_OUT / seq_id / "disp" / f"{fid}.npy"  # produced later
            row = {
                "dataset": "SERV-CT",
                "sequence_id": seq_id,
                "split": split,
                "frame_id": fid,
                "order_index": int(fid),  # numeric frame id = temporal order within experiment
                "timestamp": "",          # SERV-CT provides no timestamps
                "left_path": str((fr / "left.png").resolve()),
                "right_path": str((fr / "right.png").resolve()),
                "raw_disp_path": str(raw_disp),
                "gt_disp_path": str((fr / "disp_gt.npy").resolve()),
                "gt_depth_path": str((fr / "depth_gt_mm.npy").resolve()),
                "valid_mask_path": str((fr / "valid_mask.npy").resolve()),
                "width": calib["width"],
                "height": calib["height"],
                "disp_scale": 1.0,        # GT already px, no rescale
                "source_units": "disparity_px / depth_mm",
                "converted_units": "disparity_px / depth_mm",
                "fx_px": calib["fx"],
                "baseline_mm": calib["baseline_mm"],
                "cx_left": calib["cx_left"],
                "cy_left": calib["cy_left"],
                "reference_image": "left_rectified",
                "disp_convention": "positive_px_left_reference",
                "continuity_flag": "weak_sparse",
                "exclusion_reason": "",
            }
            manifest.append(row)
            s = seq_summ.setdefault(seq_id, {"sequence_id": seq_id, "split": split,
                                             "experiment": exp, "frames": 0, "frame_ids": []})
            s["frames"] += 1
            s["frame_ids"].append(fid)

    seq_dir_root.mkdir(parents=True, exist_ok=True)
    with (PREP / "sequence_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    summary = {
        "dataset": "SERV-CT",
        "prepared_root": str(PREP),
        "sequence_layout": str(seq_dir_root),
        "n_frames": len(manifest),
        "sequences": list(seq_summ.values()),
        "raw_disparity_recipe": {
            "generator": "scripts/temporal_refinement/data_prep/predict_s2m2_long_sequences.py",
            "variant": "S", "width": 512,
            "note": "unchanged upstream generator; output in original image disparity coords",
            "out_root": str(S2M2_OUT),
        },
        "gt_convention": "positive px, left reference, rectified (matches ARGOS; no conversion)",
        "continuity": "weak_sparse (streaming/window refiners run in causal-replay mode)",
        "non_destructive": True,
    }
    (PREP / "adapter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"n_frames": len(manifest),
                      "sequences": {k: v["frames"] for k, v in seq_summ.items()},
                      "sequence_layout": str(seq_dir_root)}, indent=2))


if __name__ == "__main__":
    build()
