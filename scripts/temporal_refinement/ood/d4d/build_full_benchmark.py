#!/usr/bin/env python3
"""Phases 2/5/6/7: aggregate the multi-specimen D4D keyframe GT into a canonical,
quality-controlled, leakage-safe benchmark.

Reads dataset/D4D/processed/keyframe_stereo_gt/keyframe_manifest.csv (ok anchors) +
conversion_rejected.csv, and produces:
  - inventory + calibration_consistency (Phase 2)
  - global quality audit (Phase 5)
  - canonical benchmark manifest + valid/warning/rejected manifests (Phase 6)
  - leakage-safe splits: session-disjoint, specimen-disjoint, LOSO, few-shot (Phase 7)

Deterministic. Rerunnable.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/dtu/p1/leopam/ARGOS")
sys.path.insert(0, str(ROOT / "scripts/temporal_refinement/ood/d4d"))
from d4d_keyframe_gt import session_root  # noqa: E402

GT = ROOT / "dataset/D4D/processed/keyframe_stereo_gt_curated"
MANIF = GT / "manifests"
SPLITS = GT / "splits"
REPORT = ROOT / "results/03_temporal_refinement/ood/d4d_full_dataset_curated"

THRESH = {"min_valid_coverage_pct": 12.0, "max_stereo_zivid_offset_ms": 60.0,
          "warn_stereo_zivid_offset_ms": 40.0, "warn_max_interp_gap_ms": 500.0,
          "disp_min_plausible": 1.0, "disp_max_plausible": 250.0}
TRANSFORM_CHAIN_VERSION = "v2-multiconv (mire45_bridge | direct_ps | direct_polaris)"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def classify(r):
    vc = float(r["valid_coverage_pct"]); off = float(r["stereo_zivid_offset_ms"])
    # curated poses have no Polaris interpolation -> no max_interp_gap_ms; treat as 0 (no gap warning).
    gap = float(r.get("max_interp_gap_ms") or 0.0); dmin = float(r["disp_min"]); dmax = float(r["disp_max"])
    if vc < THRESH["min_valid_coverage_pct"]:
        return "rejected", f"low_coverage({vc:.1f}%)"
    if off > THRESH["max_stereo_zivid_offset_ms"]:
        return "rejected", f"stereo_offset({off:.0f}ms)"
    if dmin < THRESH["disp_min_plausible"] or dmax > THRESH["disp_max_plausible"]:
        return "rejected", f"implausible_disp[{dmin:.1f},{dmax:.1f}]"
    w = []
    if off > THRESH["warn_stereo_zivid_offset_ms"]:
        w.append(f"offset>{THRESH['warn_stereo_zivid_offset_ms']:.0f}ms")
    if gap > THRESH["warn_max_interp_gap_ms"]:
        w.append(f"interp_gap>{THRESH['warn_max_interp_gap_ms']:.0f}ms")
    return ("usable_with_warning", ";".join(w)) if w else ("valid", "")


_calib_cache: dict = {}
def full_calib(specimen, session):
    key = (specimen, session)
    if key in _calib_cache:
        return _calib_cache[key]
    sdir = session_root(specimen) / session / "camera_info"
    lr = yaml.safe_load((sdir / "left_rect.yaml").read_text())
    rr = yaml.safe_load((sdir / "right_rect.yaml").read_text())
    Pl = np.array(lr["projection_matrix"]["data"]).reshape(3, 4)
    Pr = np.array(rr["projection_matrix"]["data"]).reshape(3, 4)
    ch = hashlib.md5((sdir / "left_rect.yaml").read_bytes() + (sdir / "right_rect.yaml").read_bytes()).hexdigest()[:12]
    out = {"fx": float(Pl[0, 0]), "fy": float(Pl[1, 1]), "cx": float(Pl[0, 2]), "cy": float(Pl[1, 2]),
           "baseline_m": float(-Pr[0, 3] / Pl[0, 0]), "W": lr["image_width"], "H": lr["image_height"], "hash": ch}
    _calib_cache[key] = out
    return out


def count_causal(specimen, session, stereo_ts):
    li = session_root(specimen) / session / "left_images"
    n = 0
    for p in li.glob("*.png"):
        a, b = p.stem.split("_")
        if float(f"{a}.{b}") <= stereo_ts:
            n += 1
    return n


def wcsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def main():
    global GT, MANIF, SPLITS, REPORT
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-root", type=Path, default=GT,
                    help="anchor GT root (default keyframe_stereo_gt_curated; pass keyframe_stereo_gt for the retired nominal-pose variant, if regenerated)")
    ap.add_argument("--report-root", type=Path, default=REPORT,
                    help="override report dir (default: results/.../d4d_full_dataset_curated)")
    args = ap.parse_args()
    GT = args.gt_root
    MANIF = GT / "manifests"
    SPLITS = GT / "splits"
    REPORT = args.report_root

    REPORT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((GT / "keyframe_manifest.csv").open()))
    rej_conv = list(csv.DictReader((GT / "conversion_rejected.csv").open())) if (GT / "conversion_rejected.csv").exists() else []
    commit = git_commit()

    # ---- Phase 6 canonical manifest + Phase 5 quality ----
    bench, quality, rejected = [], [], []
    for r in rows:
        cal = full_calib(r["specimen"], r["session"])
        status, reason = classify(r)
        anchor_id = f"{r['specimen']}__{r['session']}__{r['clip']}__{r['anchor']}"
        out = Path(r["out_dir"])
        rel = lambda p: str(Path(p).relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
        row = {
            "anchor_id": anchor_id, "specimen_id": r["specimen"], "session_id": r["session"],
            "clip_id": r["clip"], "anchor_type": r["anchor"], "quality_status": status, "rejection_reason": reason,
            "convention": r.get("convention", ""),
            "stereo_timestamp": r["stereo_timestamp"], "zivid_timestamp": r["zivid_timestamp"],
            "stereo_zivid_offset_ms": r["stereo_zivid_offset_ms"],
            "tracker_interpolation_interval_ms": r.get("max_interp_gap_ms", ""),
            "pose_source": r.get("pose_source", "nominal"),
            "left_rectified_path": rel(out / "left_rectified.png"), "right_rectified_path": rel(out / "right_rectified.png"),
            "gt_depth_path": rel(out / "gt_depth_left.npy"), "gt_disparity_path": rel(out / "gt_disparity_left.npy"),
            "valid_mask_path": rel(out / "valid_mask.png"), "snr_path": rel(out / "snr_mask.npy"),
            "metadata_path": rel(out / "metadata.json"),
            "fx": cal["fx"], "fy": cal["fy"], "cx": cal["cx"], "cy": cal["cy"], "baseline_m": cal["baseline_m"],
            "image_width": cal["W"], "image_height": cal["H"],
            "valid_pixel_count": r["valid_pixels"], "valid_coverage": r["valid_coverage_pct"],
            "depth_min": r["depth_min_m"], "depth_max": r["depth_max_m"],
            "disparity_min": r["disp_min"], "disparity_max": r["disp_max"],
            "calibration_hash": cal["hash"], "transform_chain_version": TRANSFORM_CHAIN_VERSION,
            "pipeline_git_commit": commit,
            "causal_frames_before": count_causal(r["specimen"], r["session"], float(r["stereo_timestamp"])),
        }
        bench.append(row)
        quality.append({"anchor_id": anchor_id, "specimen": r["specimen"], "session": r["session"],
                        "clip": r["clip"], "anchor": r["anchor"], "status": status, "reason": reason,
                        "valid_coverage_pct": r["valid_coverage_pct"], "stereo_zivid_offset_ms": r["stereo_zivid_offset_ms"],
                        "interp_gap_ms": r.get("max_interp_gap_ms", ""), "convention": r.get("convention", "")})
        if status == "rejected":
            rejected.append(quality[-1])
    # conversion-time rejects (tf/chain) into rejected list
    for r in rej_conv:
        rejected.append({"anchor_id": f"{r['specimen']}__{r['session']}__{r['clip']}__{r['anchor']}",
                         "specimen": r["specimen"], "session": r["session"], "clip": r["clip"],
                         "anchor": r["anchor"], "status": "rejected", "reason": r["reject_reason"]})

    MANIF.mkdir(parents=True, exist_ok=True)
    wcsv(MANIF / "benchmark_manifest.csv", bench)
    wcsv(MANIF / "valid_only_manifest.csv", [b for b in bench if b["quality_status"] == "valid"])
    wcsv(MANIF / "valid_and_warning_manifest.csv", [b for b in bench if b["quality_status"] in ("valid", "usable_with_warning")])
    wcsv(MANIF / "rejected_manifest.csv", [b for b in bench if b["quality_status"] == "rejected"])
    wcsv(REPORT / "global_anchor_quality.csv", quality)
    wcsv(REPORT / "global_rejected_anchors.csv", rejected)

    # ---- Phase 2 inventory + calibration consistency ----
    spec_inv, calib_cons = [], []
    by_spec = defaultdict(list)
    for b in bench:
        by_spec[b["specimen_id"]].append(b)
    for sp, items in sorted(by_spec.items()):
        sessions = sorted({i["session_id"] for i in items})
        cals = {(i["fx"], i["baseline_m"], i["convention"]) for i in items}
        spec_inv.append({"specimen": sp, "sessions": len(sessions), "anchors": len(items),
                         "valid": sum(i["quality_status"] == "valid" for i in items),
                         "warning": sum(i["quality_status"] == "usable_with_warning" for i in items),
                         "rejected": sum(i["quality_status"] == "rejected" for i in items),
                         "conventions": ",".join(sorted({i["convention"] for i in items})),
                         "fx_range": f"{min(i['fx'] for i in items):.1f}-{max(i['fx'] for i in items):.1f}",
                         "baseline_mm_range": f"{min(i['baseline_m'] for i in items)*1e3:.2f}-{max(i['baseline_m'] for i in items)*1e3:.2f}"})
        for (fx, bl, conv) in sorted(cals):
            calib_cons.append({"specimen": sp, "fx": round(fx, 2), "baseline_mm": round(bl * 1e3, 3),
                               "convention": conv, "matches_specimen_1_fx": abs(fx - 798.32) < 1.0})
    wcsv(REPORT / "specimen_inventory.csv", spec_inv)
    wcsv(REPORT / "calibration_consistency.csv", calib_cons)
    wcsv(REPORT / "session_inventory.csv",
         [{"specimen": sp, "session": s, "anchors": sum(1 for b in items if b["session_id"] == s)}
          for sp, items in by_spec.items() for s in sorted({i["session_id"] for i in items})])

    # ---- Phase 7 splits (leakage-safe: clip start+end always together; split at session/specimen) ----
    usable = [b for b in bench if b["quality_status"] in ("valid", "usable_with_warning")]
    def write_split(subdir, train, val, test, meta):
        d = SPLITS / subdir; d.mkdir(parents=True, exist_ok=True)
        wcsv(d / "train.csv", train); wcsv(d / "validation.csv", val); wcsv(d / "test.csv", test)
        (d / "split_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    rng = np.random.default_rng(0)
    sessions = sorted({(b["specimen_id"], b["session_id"]) for b in usable})
    specimens = sorted({b["specimen_id"] for b in usable})
    # session-disjoint 70/15/15 by session
    ss = list(sessions); rng.shuffle(ss)
    n = len(ss); tr, va = ss[:int(.7 * n)], ss[int(.7 * n):int(.85 * n)]; te = ss[int(.85 * n):]
    sel = lambda keys: [b for b in usable if (b["specimen_id"], b["session_id"]) in set(keys)]
    write_split("session_disjoint", sel(tr), sel(va), sel(te),
                {"strategy": "session-disjoint", "seed": 0, "train_sessions": len(tr), "val_sessions": len(va),
                 "test_sessions": len(te), "note": "all clips (start+end) of a session stay together"})
    # specimen-disjoint (if >=3 specimens)
    if len(specimens) >= 3:
        write_split("specimen_disjoint",
                    [b for b in usable if b["specimen_id"] in specimens[:-2]],
                    [b for b in usable if b["specimen_id"] == specimens[-2]],
                    [b for b in usable if b["specimen_id"] == specimens[-1]],
                    {"strategy": "specimen-disjoint", "train": specimens[:-2], "val": [specimens[-2]], "test": [specimens[-1]]})
    # leave-one-specimen-out folds
    for held in specimens:
        write_split(f"leave_one_specimen_out/hold_{held}",
                    [b for b in usable if b["specimen_id"] != held], [],
                    [b for b in usable if b["specimen_id"] == held],
                    {"strategy": "LOSO", "held_out": held, "test_anchors": sum(b["specimen_id"] == held for b in usable)})
    # few-shot (session-level, deterministic seeds)
    fs_meta = {}
    for k in (1, 2, 4, 8):
        for seed in (0, 1, 2):
            r2 = np.random.default_rng(100 + seed); ssh = list(sessions); r2.shuffle(ssh)
            train_keys = ssh[:k]; rest = ssh[k:]
            va2 = rest[:max(1, len(rest) // 4)]; te2 = rest[max(1, len(rest) // 4):]
            write_split(f"few_shot/{k}session_seed{seed}", sel(train_keys), sel(va2), sel(te2),
                        {"strategy": "few-shot", "train_sessions": k, "seed": seed})
        fs_meta[f"{k}session"] = {"seeds": [0, 1, 2]}
    for frac in (0.1, 0.25, 0.5):
        r3 = np.random.default_rng(200); ssh = list(sessions); r3.shuffle(ssh)
        k = max(1, int(len(ssh) * frac)); train_keys = ssh[:k]; rest = ssh[k:]
        va3 = rest[:max(1, len(rest) // 4)]; te3 = rest[max(1, len(rest) // 4):]
        write_split(f"few_shot/frac{int(frac*100)}pct", sel(train_keys), sel(va3), sel(te3),
                    {"strategy": "few-shot-fraction", "fraction": frac, "train_sessions": k})

    # ---- summaries ----
    cov = [float(b["valid_coverage"]) for b in usable]
    off = [float(b["stereo_zivid_offset_ms"]) for b in usable]
    status_counts = Counter(b["quality_status"] for b in bench)
    summary = {
        "git_commit": commit, "transform_chain_version": TRANSFORM_CHAIN_VERSION,
        "specimens_processed": specimens, "n_sessions": len(sessions),
        "total_ok_anchors": len(bench), "status_counts": dict(status_counts),
        "benchmark_usable": len(usable),
        "conversion_rejected_tf_chain": len(rej_conv),
        "coverage_pct": {"min": round(min(cov), 1), "median": round(float(np.median(cov)), 1), "max": round(max(cov), 1)},
        "stereo_zivid_offset_ms": {"min": round(min(off), 1), "median": round(float(np.median(off)), 1), "max": round(max(off), 1)},
        "conventions": dict(Counter(b["convention"] for b in bench)),
        "per_specimen": {sp: {"anchors": len(v), "valid": sum(x["quality_status"] == "valid" for x in v),
                              "warning": sum(x["quality_status"] == "usable_with_warning" for x in v)}
                         for sp, v in by_spec.items()},
    }
    (REPORT / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (REPORT / "split_summary.json").write_text(json.dumps({
        "splits": [str(p.relative_to(SPLITS)) for p in SPLITS.rglob("split_metadata.json")],
        "usable_anchors": len(usable)}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
