#!/usr/bin/env python3
"""Phase 5+8: anchor quality filtering + eval-ready benchmark manifest for D4D keyframe GT.

Reads the per-anchor keyframe_manifest.csv, applies transparent objective quality thresholds,
classifies anchors (valid / usable_with_warning / rejected), and emits the benchmark manifest
+ quality reports under results/03_temporal_refinement/ood/d4d_keyframe_gt/.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

GT_ROOT = Path("/dtu/p1/leopam/ARGOS/dataset/D4D/processed/keyframe_stereo_gt")
REPORT = Path("/dtu/p1/leopam/ARGOS/results/03_temporal_refinement/ood/d4d_keyframe_gt")

THRESH = {
    "min_valid_coverage_pct": 12.0,
    "max_stereo_zivid_offset_ms": 60.0,
    "warn_stereo_zivid_offset_ms": 40.0,
    "warn_max_interp_gap_ms": 500.0,
    "disp_min_plausible": 1.0,
    "disp_max_plausible": 250.0,
}


def classify(r: dict) -> tuple[str, str]:
    vc = float(r["valid_coverage_pct"]); off = float(r["stereo_zivid_offset_ms"])
    gap = float(r["max_interp_gap_ms"]); dmin = float(r["disp_min"]); dmax = float(r["disp_max"])
    if vc < THRESH["min_valid_coverage_pct"]:
        return "rejected", f"low_coverage({vc:.1f}%)"
    if off > THRESH["max_stereo_zivid_offset_ms"]:
        return "rejected", f"stereo_offset({off:.0f}ms)"
    if dmin < THRESH["disp_min_plausible"] or dmax > THRESH["disp_max_plausible"]:
        return "rejected", f"implausible_disp[{dmin:.1f},{dmax:.1f}]"
    warns = []
    if off > THRESH["warn_stereo_zivid_offset_ms"]:
        warns.append(f"offset>{THRESH['warn_stereo_zivid_offset_ms']:.0f}ms")
    if gap > THRESH["warn_max_interp_gap_ms"]:
        warns.append(f"interp_gap>{THRESH['warn_max_interp_gap_ms']:.0f}ms")
    return ("usable_with_warning", ";".join(warns)) if warns else ("valid", "")


def count_causal_frames(out_dir: str, stereo_ts: float) -> int:
    # available causal (<=anchor) stereo frames in the session
    session = Path(out_dir).parents[2]  # .../session/clip/anchor -> session
    raw_session = Path("/dtu/p1/leopam/ARGOS/dataset/D4D/raw/extracted/specimen_1/specimen_1") / session.name
    lefts = glob.glob(str(raw_session / "left_images/*.png"))
    n = 0
    for p in lefts:
        a, b = Path(p).stem.split("_")
        if float(f"{a}.{b}") <= stereo_ts:
            n += 1
    return n


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((GT_ROOT / "keyframe_manifest.csv").open()))
    quality, rejected, bench = [], [], []
    counts = {"valid": 0, "usable_with_warning": 0, "rejected": 0}
    for r in rows:
        status, reason = classify(r)
        counts[status] += 1
        qrow = {"specimen": r["specimen"], "session": r["session"], "clip": r["clip"], "anchor": r["anchor"],
                "status": status, "reason": reason, "valid_coverage_pct": r["valid_coverage_pct"],
                "stereo_zivid_offset_ms": r["stereo_zivid_offset_ms"], "max_interp_gap_ms": r["max_interp_gap_ms"],
                "photometric_rgb_mae": r["photometric_rgb_mae"], "disp_min": r["disp_min"], "disp_max": r["disp_max"]}
        quality.append(qrow)
        if status == "rejected":
            rejected.append(qrow); continue
        out = r["out_dir"]
        bench.append({
            "specimen": r["specimen"], "session": r["session"], "clip": r["clip"], "anchor": r["anchor"],
            "status": status,
            "stereo_frame": r["stereo_frame"], "stereo_timestamp": r["stereo_timestamp"],
            "zivid_timestamp": r["zivid_timestamp"], "time_offset_ms": r["stereo_zivid_offset_ms"],
            "left_rectified": str(Path(out) / "left_rectified.png"),
            "right_rectified": str(Path(out) / "right_rectified.png"),
            "gt_depth": str(Path(out) / "gt_depth_left.npy"),
            "gt_disparity": str(Path(out) / "gt_disparity_left.npy"),
            "valid_mask": str(Path(out) / "valid_mask.png"),
            "snr_mask": str(Path(out) / "snr_mask.npy"),
            "fx_px": r["fx_px"], "baseline_mm": r["baseline_mm"], "resolution": r["resolution"],
            "valid_pixels": r["valid_pixels"], "valid_coverage_pct": r["valid_coverage_pct"],
            "n_causal_frames": count_causal_frames(out, float(r["stereo_timestamp"])),
        })

    def wcsv(path, data):
        if not data:
            path.write_text(""); return
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)

    wcsv(REPORT / "anchor_quality.csv", quality)
    wcsv(REPORT / "rejected_anchors.csv", rejected)
    wcsv(REPORT / "benchmark_manifest.csv", bench)
    (REPORT / "quality_thresholds.json").write_text(json.dumps(THRESH, indent=2) + "\n")
    summary = {"total_converted_anchors": len(rows), **counts,
               "benchmark_usable": counts["valid"] + counts["usable_with_warning"],
               "note": "27 additional candidate anchors failed conversion (missing tf series or snr file); see audit log"}
    (REPORT / "quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
