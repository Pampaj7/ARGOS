#!/usr/bin/env python3
"""Phase 1 forensic audit of D4D formats. Read-only. Writes the audit report CSVs/JSON.

Documents exact depth/pointcloud/tf/calibration/clip formats and timestamp coverage.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np
import yaml

RAW = Path("/dtu/p1/leopam/ARGOS/dataset/D4D/raw/extracted")
OUT = Path("/dtu/p1/leopam/ARGOS/results/03_temporal_refinement/ood/d4d_keyframe_gt_audit")


def sessions(specimen="specimen_1"):
    inner = RAW / specimen / specimen
    return [s for s in sorted(inner.glob("*")) if s.is_dir()]


def ts(name, kind="tf"):
    s = Path(name).stem
    if kind == "tf":
        return float(s.rsplit("_", 1)[1])
    a, b = s.split("_"); return float(f"{a}.{b}")


def wcsv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sess = sessions()
    s0 = sess[0]

    # dataset_structure.json
    struct = {"root": str(RAW), "specimen_1_sessions": len(sess),
              "session_subdirs": sorted([p.name for p in s0.glob("*") if p.is_dir()]),
              "hierarchy": "dataset -> specimen_N -> session_timestamp -> {stereo stream, camera_info, zivid geometry (2/clip), tf/, clips.json}"}
    (OUT / "dataset_structure.json").write_text(json.dumps(struct, indent=2) + "\n")

    # depth_npy_format.csv (sample first anchor of a few sessions)
    depth_rows, pc_rows = [], []
    for s in sess[:4]:
        for dp in sorted(glob.glob(str(s / "depth_images/*.npy")))[:1]:
            d = np.load(dp); f = np.isfinite(d)
            depth_rows.append({"session": s.name, "file": Path(dp).name, "shape": str(d.shape), "dtype": str(d.dtype),
                               "finite_pct": round(100 * f.mean(), 2), "min_m": round(float(np.nanmin(d[f])), 4),
                               "max_m": round(float(np.nanmax(d[f])), 4), "p50_m": round(float(np.nanpercentile(d[f], 50)), 4),
                               "invalid": "NaN", "units": "metres (optical-axis Z)", "frame": "zivid_optical (color cam aligned)",
                               "resolution_note": "2448x2048 = Zivid color resolution"})
        for pp in sorted(glob.glob(str(s / "pointcloud/*.ply")))[:1]:
            with open(pp, "rb") as fh:
                hdr = []
                for _ in range(30):
                    ln = fh.readline().decode("ascii", "replace").strip(); hdr.append(ln)
                    if ln == "end_header":
                        break
            n = next((int(l.split()[-1]) for l in hdr if l.startswith("element vertex")), None)
            pc_rows.append({"session": s.name, "file": Path(pp).name, "format": "binary_little_endian 1.0 (Open3D)",
                            "vertices": n, "properties": "double x,y,z + uchar r,g,b", "units": "metres",
                            "frame": "zivid_optical_frame", "equals_depth_backprojection": True})
    wcsv(OUT / "depth_npy_format.csv", depth_rows)
    wcsv(OUT / "pointcloud_format.csv", pc_rows)

    # calibration_fields.csv
    cal_rows = []
    ci = s0 / "camera_info"
    for n in ["left.yaml", "right.yaml", "left_rect.yaml", "right_rect.yaml", "color_camera_info.yaml"]:
        d = yaml.safe_load((ci / n).read_text())
        K = np.array(d.get("camera_matrix", {}).get("data", d.get("K"))).reshape(3, 3)
        P = d.get("projection_matrix", {}).get("data", d.get("P"))
        P = np.array(P).reshape(3, 4) if P else None
        cal_rows.append({"file": n, "frame": d.get("camera_name"), "W": d.get("image_width"), "H": d.get("image_height"),
                         "distortion_model": d.get("distortion_model"), "fx": round(float(K[0, 0]), 3), "cx": round(float(K[0, 2]), 3),
                         "Tx_P03": round(float(P[0, 3]), 4) if P is not None else None,
                         "R_is_identity": bool(np.allclose(np.array(d.get("rectification_matrix", {}).get("data", d.get("R", np.eye(3).ravel()))).reshape(3, 3), np.eye(3)))})
    # baseline
    lr = yaml.safe_load((ci / "left_rect.yaml").read_text()); rr = yaml.safe_load((ci / "right_rect.yaml").read_text())
    fx = np.array(lr["projection_matrix"]["data"]).reshape(3, 4)[0, 0]
    Tx = np.array(rr["projection_matrix"]["data"]).reshape(3, 4)[0, 3]
    cal_rows.append({"file": "DERIVED", "frame": "stereo_rectified", "W": lr["image_width"], "H": lr["image_height"],
                     "distortion_model": "-", "fx": round(float(fx), 3), "cx": None, "Tx_P03": round(float(Tx), 4),
                     "R_is_identity": f"baseline_mm={-Tx/fx*1000:.3f}"})
    wcsv(OUT / "calibration_fields.csv", cal_rows)

    # tf_format.csv
    tf_types = {}
    for p in glob.glob(str(s0 / "tf/*.json")):
        pre = Path(p).name.rsplit("_", 1)[0]
        tf_types.setdefault(pre, p)
    tf_rows = []
    for pre, p in sorted(tf_types.items()):
        d = json.loads(Path(p).read_text())
        tf_rows.append({"prefix": pre, "count": len(glob.glob(str(s0 / f"tf/{pre}_*.json"))),
                        "parent_frame": d["parent_frame"], "child_frame": d["child_frame"],
                        "schema": "timestamp, parent_frame, child_frame, transform{translation xyz, rotation xyzw}",
                        "direction": "T maps child->parent (p_parent = T @ p_child); translation = child origin in parent",
                        "units": "metres; quaternion xyzw; right-handed",
                        "applied_calibration": d.get("applied_calibration")})
    wcsv(OUT / "tf_format.csv", tf_rows)

    # timestamp_analysis.csv + clip_geometry_mapping.csv
    tsr, clipr = [], []
    for s in sess:
        stereo = [ts(Path(p).name, "stereo") for p in glob.glob(str(s / "left_images/*.png"))]
        if not stereo:
            continue
        zivid = [ts(Path(p).name, "z") for p in glob.glob(str(s / "depth_images/*.npy"))]
        cam = [ts(p) for p in glob.glob(str(s / "tf/polaris_spectra_to_camera_optical_*.json"))]
        ziv = [ts(p) for p in glob.glob(str(s / "tf/polaris_to_zivid_optical_frame_*.json"))]
        row = {"session": s.name, "n_stereo": len(stereo), "n_zivid_scans": len(zivid),
               "stereo_span_s": round(max(stereo) - min(stereo), 2), "n_cam_tf": len(cam), "n_zivid_tf": len(ziv),
               "stereo_fps": round(len(stereo) / max(max(stereo) - min(stereo), 1e-6), 1)}
        offs = []
        for z in zivid:
            offs.append(min(abs(z - x) for x in stereo) * 1e3)
        row["max_stereo_zivid_offset_ms"] = round(max(offs), 1) if offs else None
        tsr.append(row)
        cj = s / "clips.json"
        if cj.exists():
            for c in json.loads(cj.read_text()).get("clips", []):
                for k in ("start", "end"):
                    g = c.get(f"{k}_geometry")
                    stem = Path(g).stem if g else None
                    exists = bool(stem and (s / "depth_images" / f"{stem}.npy").exists())
                    clipr.append({"session": s.name, "clip": c.get("name"), "anchor": k,
                                  "geometry": g, "zivid_stem": stem, "depth_exists": exists})
    wcsv(OUT / "timestamp_analysis.csv", tsr)
    wcsv(OUT / "clip_geometry_mapping.csv", clipr)
    print(f"audit written: {len(sess)} sessions, {len(clipr)} clip-anchors, {sum(r['depth_exists'] for r in clipr)} with depth")


if __name__ == "__main__":
    main()
