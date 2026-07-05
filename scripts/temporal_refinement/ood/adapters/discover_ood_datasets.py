#!/usr/bin/env python3
"""Phase 1 OOD dataset discovery: inventory D4D + SERV-CT non-destructively.

Reads only. Emits inventory CSV/JSON to
results/03_temporal_refinement/ood/dataset_discovery/.

Agent B / OOD benchmark. Does not touch raw source datasets or any model files.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path("/dtu/p1/leopam/ARGOS")
OUT = ROOT / "results/03_temporal_refinement/ood/dataset_discovery"

SERVCT_ARGOS = ROOT / "dataset/SERVCT/argos/servct_argos"
SERVCT_RAW = ROOT / "dataset/SERVCT/raw/extracted/SERV-CT"
D4D_EXTRACTED = ROOT / "dataset/D4D/raw/extracted"


def arr_stats(path: Path) -> dict:
    a = np.load(path).astype(np.float32)
    f = np.isfinite(a)
    valid = f & (a != 0)
    return {
        "shape": list(a.shape),
        "dtype": str(np.load(path).dtype),
        "min": float(np.nanmin(a[f])) if f.any() else None,
        "max": float(np.nanmax(a[f])) if f.any() else None,
        "mean": float(np.nanmean(a[f])) if f.any() else None,
        "finite_pct": round(100.0 * f.mean(), 2),
        "nonzero_finite_pct": round(100.0 * valid.mean(), 2),
    }


def discover_servct() -> tuple[list[dict], dict]:
    rows = []
    seqs: dict[tuple, list] = {}
    for split_dir in sorted(SERVCT_ARGOS.glob("*")):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for fr in sorted(split_dir.glob("*")):
            if not fr.is_dir():
                continue
            meta = json.loads((fr / "metadata.json").read_text())
            calib = json.loads((fr / "calib.json").read_text())
            disp = arr_stats(fr / "disp_gt.npy")
            depth = arr_stats(fr / "depth_gt_mm.npy")
            vm = np.load(fr / "valid_mask.npy")
            exp = meta["sequence"]
            seqs.setdefault((split, exp), []).append(meta["frame"])
            rows.append({
                "dataset": "SERV-CT",
                "split": split,
                "sequence_id": exp,
                "frame_id": meta["frame"],
                "reference_type": meta.get("reference_type"),
                "width": calib["width"],
                "height": calib["height"],
                "fx_px": calib["fx"],
                "baseline_mm": calib["baseline_mm"],
                "has_stereo": (fr / "left.png").exists() and (fr / "right.png").exists(),
                "has_raw_disp": False,  # S2M2 not yet run OOD
                "has_gt_disp": (fr / "disp_gt.npy").exists(),
                "has_gt_depth": (fr / "depth_gt_mm.npy").exists(),
                "has_valid_mask": (fr / "valid_mask.npy").exists(),
                "gt_disp_min_px": disp["min"],
                "gt_disp_max_px": disp["max"],
                "gt_disp_mean_px": disp["mean"],
                "gt_depth_min_mm": depth["min"],
                "gt_depth_max_mm": depth["max"],
                "valid_mask_pct": round(100.0 * (vm > 0).mean(), 2),
                "gt_density": "dense",
                "disp_convention": "positive_px_left_reference_rectified",
                "units": meta.get("units"),
            })
    summary = {
        "root": str(SERVCT_ARGOS),
        "raw_source": str(SERVCT_RAW),
        "n_frames": len(rows),
        "sequences": {f"{s}/{e}": sorted(f) for (s, e), f in seqs.items()},
        "n_sequences": len(seqs),
        "gt": "dense disparity (px) + depth (mm) + valid mask, aligned to LEFT rectified image",
        "rectified": True,
        "disp_convention": "positive pixel disparity, left reference (matches ARGOS/S2M2)",
        "temporal": "each Experiment is an ordered set of pairs; continuity WEAK/SPARSE",
        "raw_disparity_status": "ABSENT — pretrained S2M2-S inference required (zero-shot upstream)",
        "license": "SERV-CT (Edwards et al. 2020), academic use; cite original",
    }
    return rows, summary


def _d4d_calib(session: Path) -> dict:
    out = {}
    ci = session / "camera_info"
    for side in ("left", "right"):
        y = ci / f"{side}_rect.yaml"
        if not y.exists():
            continue
        d = yaml.safe_load(y.read_text())
        P = np.array(d["projection_matrix"]["data"]).reshape(3, 4)
        out[side] = P
        out[f"{side}_w"] = d.get("image_width")
        out[f"{side}_h"] = d.get("image_height")
    res = {}
    if "left" in out:
        res["fx_px"] = float(out["left"][0, 0])
        res["cx"] = float(out["left"][0, 2])
        res["width"] = out.get("left_w")
        res["height"] = out.get("left_h")
    if "left" in out and "right" in out:
        fx = out["left"][0, 0]
        # Tx = P_right[0,3] = -fx * baseline  ->  baseline = -Tx / fx
        res["baseline_m"] = float(-out["right"][0, 3] / fx) if fx else None
    return res


def discover_d4d() -> tuple[list[dict], dict]:
    rows = []
    total_frames = 0
    total_clips = 0
    specimens_present = []
    for spec in sorted(D4D_EXTRACTED.glob("specimen_*")):
        inner = spec / spec.name
        if not inner.is_dir():
            continue  # e.g. 'info' or non-standard extraction
        specimens_present.append(spec.name)
        for session in sorted(inner.glob("*")):
            if not session.is_dir():
                continue
            li = session / "left_images"
            ri = session / "right_images"
            di = session / "depth_images"
            pc = session / "pointcloud"
            n_left = len(list(li.glob("*.png"))) + len(list(li.glob("*.jpg"))) if li.exists() else 0
            n_right = len(list(ri.glob("*.png"))) + len(list(ri.glob("*.jpg"))) if ri.exists() else 0
            n_depth = len(list(di.glob("*"))) if di.exists() else 0
            n_pc = len(list(pc.glob("*"))) if pc.exists() else 0
            n_clips = 0
            cj = session / "clips.json"
            if cj.exists():
                try:
                    n_clips = len(json.loads(cj.read_text()).get("clips", []))
                except Exception:
                    n_clips = -1
            res = {"width": None, "height": None}
            if n_left:
                sample = sorted(li.glob("*.png")) or sorted(li.glob("*.jpg"))
                with Image.open(sample[0]) as im:
                    res["width"], res["height"] = im.size
            calib = _d4d_calib(session)
            total_frames += n_left
            total_clips += max(n_clips, 0)
            rows.append({
                "dataset": "D4D",
                "specimen": spec.name,
                "session_id": session.name,
                "n_left_images": n_left,
                "n_right_images": n_right,
                "n_depth_images_zivid_gt": n_depth,
                "n_pointclouds_zivid_gt": n_pc,
                "n_clips": n_clips,
                "img_width": res["width"],
                "img_height": res["height"],
                "rect_fx_px": calib.get("fx_px"),
                "rect_baseline_m": calib.get("baseline_m"),
                "has_stereo": n_left > 0 and n_right > 0,
                "has_raw_disp": False,
                "has_dense_pergframe_gt": False,
                "gt_type": "Zivid structured-light (sparse in time: per-scan, not per-frame)",
                "rectification_state": "raw images + rectification params (left_rect/right_rect yaml) present; rect NOT pre-applied",
            })
    summary = {
        "root": str(D4D_EXTRACTED),
        "specimens_extracted": specimens_present,
        "specimens_not_extracted": ["specimen_3", "specimen_4", "specimen_5 (tar.gz present, not extracted)"],
        "n_sessions": len(rows),
        "n_frames_total_approx": total_frames,
        "n_clips_total": total_clips,
        "gt": "Zivid structured-light scans (depth_images + pointcloud), ~2 per session at scan timepoints",
        "dense_perframe_disp_gt": False,
        "rectified": "params available (R,P per side); must apply cv2.remap",
        "disp_convention": "after rectification: positive px left reference (to be produced)",
        "temporal": "true temporal video within clips (ordered frames, ~fps continuity)",
        "raw_disparity_status": "ABSENT — pretrained S2M2-S inference required",
        "license": "D4D Dresden 4D (Wagner et al.), academic use; cite original",
        "blocker": "no dense per-frame GT disparity — requires Zivid pointcloud reprojection + per-frame pose to build GT",
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    serv_rows, serv_sum = discover_servct()
    d4d_rows, d4d_sum = discover_d4d()
    write_csv(OUT / "servct_inventory.csv", serv_rows)
    write_csv(OUT / "d4d_inventory.csv", d4d_rows)

    # Standardized candidate sequence manifest (both datasets)
    manifest = []
    # SERV-CT: one candidate sequence per experiment
    serv_seq_frames: dict = {}
    for r in serv_rows:
        serv_seq_frames.setdefault((r["split"], r["sequence_id"]), []).append(r["frame_id"])
    for (split, exp), fr in sorted(serv_seq_frames.items()):
        manifest.append({
            "dataset": "SERV-CT",
            "sequence_id": f"{split}/{exp}",
            "n_frames": len(fr),
            "has_stereo": True,
            "has_raw_disp": False,
            "has_gt": True,
            "gt_type": "dense_disp_px",
            "temporal_continuity": "weak_sparse",
            "usable_zero_shot_singleframe": True,
            "usable_zero_shot_streaming": False,
            "exclusion_reason": "" if split == "honest_test" else "train split (still zero-shot but reserve as OOD holdout; not used for tuning)",
        })
    # D4D: one candidate sequence per session (frames temporally ordered)
    for r in d4d_rows:
        usable_gt = r["n_depth_images_zivid_gt"] > 0
        manifest.append({
            "dataset": "D4D",
            "sequence_id": f"{r['specimen']}/{r['session_id']}",
            "n_frames": r["n_left_images"],
            "has_stereo": r["has_stereo"],
            "has_raw_disp": False,
            "has_gt": usable_gt,
            "gt_type": "zivid_sparse_scan",
            "temporal_continuity": "strong_video",
            "usable_zero_shot_singleframe": False,
            "usable_zero_shot_streaming": False,
            "exclusion_reason": "no dense per-frame GT disparity; needs Zivid reprojection+pose conversion before benchmark",
        })
    write_csv(OUT / "candidate_sequence_manifest.csv", manifest)

    discovered = {
        "generated_by": "scripts/temporal_refinement/ood/adapters/discover_ood_datasets.py",
        "purpose": "Zero-shot OOD benchmark for ARGOS temporal stereo refiners (Agent B)",
        "argos_refiner_upstream": "S2M2-S @ 512 (pretrained, zero-shot) — refiners correct this raw disparity",
        "datasets": {"SERV-CT": serv_sum, "D4D": d4d_sum},
        "headline": {
            "servct": "READY for zero-shot after S2M2-S raw-disp generation; dense GT; weak temporal.",
            "d4d": "NOT READY — no dense per-frame GT disparity; heavy Zivid->disp conversion required.",
        },
    }
    (OUT / "discovered_datasets.json").write_text(json.dumps(discovered, indent=2) + "\n")
    print(json.dumps({"servct_frames": len(serv_rows), "d4d_sessions": len(d4d_rows),
                      "d4d_frames_approx": d4d_sum["n_frames_total_approx"],
                      "manifest_rows": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
