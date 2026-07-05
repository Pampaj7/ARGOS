#!/usr/bin/env python3
"""D4D -> ARGOS OOD adapter (non-destructive inventory / manifest).

D4D is a true temporal-video surgical dataset, but its only ground truth is Zivid
structured-light captured at ~2 scan timepoints per session — there is NO dense
per-frame disparity GT (see dataset_discovery/missing_requirements.md, blocker B2).
It also ships UNRECTIFIED stereo (rectification params present but not applied).

This adapter therefore does the honest, non-fabricating thing:
  * enumerates specimen/session/clip structure and temporal frame order,
  * records calibration (rectified P matrices, fx, baseline) and continuity,
  * emits a sequence manifest with has_gt=False and an explicit exclusion_reason,
  * does NOT invent disparity GT and does NOT run S2M2 (raw disparity on unrectified
    pairs would be meaningless).

To promote D4D to a real benchmark you must first: (a) rectify with the yaml R/P via
cv2.initUndistortRectifyMap, (b) build GT by Zivid pointcloud reprojection + per-frame
pose. Both are documented in missing_requirements.md. This adapter stops before them.

Outputs under: results/03_temporal_refinement/ood/prepared/d4d/
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path("/dtu/p1/leopam/ARGOS")
SRC = ROOT / "dataset/D4D/raw/extracted"
PREP = ROOT / "results/03_temporal_refinement/ood/prepared/d4d"


def rect_calib(session: Path) -> dict:
    ci = session / "camera_info"
    out = {}
    P = {}
    for side in ("left", "right"):
        y = ci / f"{side}_rect.yaml"
        if y.exists():
            d = yaml.safe_load(y.read_text())
            P[side] = np.array(d["projection_matrix"]["data"]).reshape(3, 4)
            out[f"{side}_w"] = d.get("image_width"); out[f"{side}_h"] = d.get("image_height")
    if "left" in P:
        out["fx_px"] = float(P["left"][0, 0]); out["cx"] = float(P["left"][0, 2])
    if "left" in P and "right" in P and P["left"][0, 0]:
        out["baseline_m"] = float(-P["right"][0, 3] / P["left"][0, 0])
    return out


def build() -> None:
    PREP.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    seqs: list[dict] = []
    for spec in sorted(SRC.glob("specimen_*")):
        inner = spec / spec.name
        if not inner.is_dir():
            continue
        for session in sorted(inner.glob("*")):
            if not session.is_dir():
                continue
            li = session / "left_images"; ri = session / "right_images"
            lefts = sorted(li.glob("*.png")) if li.exists() else []
            rights = sorted(ri.glob("*.png")) if ri.exists() else []
            if not lefts or not rights:
                continue
            calib = rect_calib(session)
            res = None
            with Image.open(lefts[0]) as im:
                res = im.size
            sid = f"{spec.name}__{session.name}"
            # pair by index (stems assumed aligned); count paired frames
            n = min(len(lefts), len(rights))
            seqs.append({"sequence_id": sid, "specimen": spec.name, "session": session.name,
                         "n_frames": n, "img_w": res[0], "img_h": res[1],
                         "fx_px": calib.get("fx_px"), "baseline_m": calib.get("baseline_m")})
            for i in range(n):
                manifest.append({
                    "dataset": "D4D", "sequence_id": sid, "frame_id": lefts[i].stem,
                    "order_index": i, "timestamp": lefts[i].stem,
                    "left_path": str(lefts[i]), "right_path": str(rights[i]),
                    "raw_disp_path": "", "gt_disp_path": "", "gt_depth_path": "",
                    "valid_mask_path": "",
                    "width": res[0], "height": res[1],
                    "fx_px": calib.get("fx_px"), "baseline_m": calib.get("baseline_m"),
                    "reference_image": "left_UNRECTIFIED",
                    "disp_convention": "n/a (rectification not applied)",
                    "continuity_flag": "strong_video",
                    "has_gt": False,
                    "exclusion_reason": "B2: no dense per-frame disparity GT (Zivid sparse) + rectification not applied",
                })
    if manifest:
        with (PREP / "sequence_manifest.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            w.writeheader(); w.writerows(manifest)
    (PREP / "adapter_summary.json").write_text(json.dumps({
        "dataset": "D4D", "prepared_root": str(PREP), "n_sequences": len(seqs),
        "n_frames": len(manifest), "sequences": seqs,
        "status": "INVENTORY ONLY — excluded from dense zero-shot disparity benchmark",
        "blockers": ["B2 no dense per-frame disparity GT (Zivid sparse)",
                     "rectification params present but not applied"],
        "non_destructive": True,
        "future_use": "GT-free temporal-stability OOD diagnostics after rectification, or "
                      "few-frame GT eval at Zivid scan instants (see missing_requirements.md)",
    }, indent=2) + "\n")
    print(json.dumps({"n_sequences": len(seqs), "n_frames": len(manifest),
                      "status": "inventory only (D4D excluded from dense benchmark; blocker B2)"}, indent=2))


if __name__ == "__main__":
    build()
