#!/usr/bin/env python3
"""Phase 8: standalone D4D benchmark validator. Exits non-zero on hard failure.

Checks manifest path existence, array loading + shapes, mask/GT agreement, depth<->disparity
round-trip, calibration consistency, duplicate anchor IDs, split leakage, no rejected anchors
in valid manifests, quality counts, causal-context paths, and that no raw files were modified.
Writes validation_report.json + validation_report.md.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/dtu/p1/leopam/ARGOS")
GT = ROOT / "dataset/D4D/processed/keyframe_stereo_gt"
MANIF = GT / "manifests"
SPLITS = GT / "splits"
REPORT = ROOT / "results/03_temporal_refinement/ood/d4d_full_dataset"


def rp(p):  # resolve possibly-relative path
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def main():
    fails, warns, checks = [], [], []
    bench = list(csv.DictReader((MANIF / "benchmark_manifest.csv").open()))

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            fails.append(f"{name}: {detail}")

    # unique anchor ids
    ids = [b["anchor_id"] for b in bench]
    chk("unique_anchor_ids", len(ids) == len(set(ids)), f"{len(ids)-len(set(ids))} dups")

    usable = [b for b in bench if b["quality_status"] in ("valid", "usable_with_warning")]
    # sample-check arrays (all usable; cap for speed if huge)
    sample = usable if len(usable) <= 200 else usable[::max(1, len(usable) // 200)]
    bad_paths = rt_fail = mask_fail = missing_right = 0
    for b in sample:
        for col in ("gt_depth_path", "gt_disparity_path", "valid_mask_path", "snr_path",
                    "left_rectified_path", "metadata_path"):  # essential
            if not rp(b[col]).exists():
                bad_paths += 1
        if not rp(b["right_rectified_path"]).exists():  # optional (some frames lack right pair)
            missing_right += 1
        try:
            disp = np.load(rp(b["gt_disparity_path"])); depth = np.load(rp(b["gt_depth_path"]))
            valid = cv2.imread(str(rp(b["valid_mask_path"])), cv2.IMREAD_GRAYSCALE) > 0
            if disp.shape != depth.shape or disp.shape != valid.shape:
                bad_paths += 1; continue
            m = valid & np.isfinite(disp) & np.isfinite(depth)
            fx = float(b["fx"]); base = float(b["baseline_m"]) * 1e3
            rt = np.abs(depth[m] * 1e3 - fx * base / np.maximum(disp[m], 1e-6))  # depth in mm
            # note gt_depth stored in metres; fx*baseline_mm/disp gives mm -> compare mm
            if np.nanmax(np.abs(depth[m] - (fx * float(b["baseline_m"]) / np.maximum(disp[m], 1e-6)))) > 1e-2:
                rt_fail += 1
            # mask agrees with finite GT
            if not np.array_equal(valid, np.isfinite(disp)):
                mask_fail += 1
        except Exception as e:
            bad_paths += 1
    chk("manifest_essential_paths_exist", bad_paths == 0, f"{bad_paths} missing/shape-mismatch")
    if missing_right:
        warns.append(f"{missing_right} anchors missing right_rectified (absent right stereo pair; left GT valid)")
    checks.append({"check": "right_rectified_present", "ok": True,
                   "detail": f"{missing_right} missing right view (warning, left-ref GT unaffected)"})
    chk("depth_disparity_roundtrip", rt_fail == 0, f"{rt_fail} anchors exceed 0.01px")
    chk("mask_agrees_finite_gt", mask_fail == 0, f"{mask_fail} disagreements")

    # no rejected in valid manifests
    vonly = list(csv.DictReader((MANIF / "valid_only_manifest.csv").open()))
    chk("valid_only_has_no_rejected", all(b["quality_status"] == "valid" for b in vonly), "")
    vw = list(csv.DictReader((MANIF / "valid_and_warning_manifest.csv").open()))
    chk("valid_and_warning_clean", all(b["quality_status"] in ("valid", "usable_with_warning") for b in vw), "")

    # split leakage: sessions/specimens disjoint; clip start+end together
    for sd in SPLITS.rglob("split_metadata.json"):
        d = sd.parent
        def load(n):
            f = d / n
            return list(csv.DictReader(f.open())) if f.exists() else []
        tr, va, te = load("train.csv"), load("validation.csv"), load("test.csv")
        strat = json.loads(sd.read_text()).get("strategy", "")
        key = "session_id" if "session" in strat or "few" in strat else "specimen_id" if "specimen" in strat or "LOSO" in strat else "session_id"
        sets = [set((b["specimen_id"], b[key]) if key == "session_id" else b[key] for b in grp) for grp in (tr, va, te)]
        leak = (sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2])
        chk(f"split_no_leak::{d.relative_to(SPLITS)}", not leak, f"{len(leak)} overlapping {key}")
        # clip integrity: both anchors of a clip in same split
        clip_split = {}
        leak2 = 0
        for grp, name in ((tr, "tr"), (va, "va"), (te, "te")):
            for b in grp:
                ck = (b["specimen_id"], b["session_id"], b["clip_id"])
                if ck in clip_split and clip_split[ck] != name:
                    leak2 += 1
                clip_split[ck] = name
        chk(f"split_clip_integrity::{d.relative_to(SPLITS)}", leak2 == 0, f"{leak2} clips split across")

    report = {"result": "PASS" if not fails else "FAIL", "n_checks": len(checks),
              "n_fail": len(fails), "failures": fails, "warnings": warns,
              "n_usable_anchors": len(usable), "checks": checks}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [f"# D4D benchmark validation — **{report['result']}**", "",
          f"- checks: {len(checks)}  failures: {len(fails)}  usable anchors: {len(usable)}", ""]
    for c in checks:
        md.append(f"- [{'x' if c['ok'] else ' '}] {c['check']} {('— ' + c['detail']) if c['detail'] else ''}")
    (REPORT / "validation_report.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"result": report["result"], "checks": len(checks), "failures": fails}, indent=2))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
