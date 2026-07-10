#!/usr/bin/env python3
"""Validate the pilot cache against the promotion criteria: integrity, timing, storage
projection, float16 quantization error, random mmap access, and visual contact sheets.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from argos_v2.backbones import BACKBONE_NAMES
from argos_v2.paths import CACHE_DIR, CACHE_HEIGHT, CACHE_WIDTH, QUALITY_GATE_CSV, RESULTS_DIR
from argos_v2.scared_c_data import load_frame_gt, load_frame_lr, load_sequence_info
from run_pilot import PILOT_SEQUENCES

OUT_DIR = RESULTS_DIR / "pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONTACT_DIR = OUT_DIR / "contact_sheets"
CONTACT_DIR.mkdir(parents=True, exist_ok=True)


def dir_size_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def cache_integrity() -> list[dict]:
    rows = []
    for backbone in BACKBONE_NAMES:
        for seq in PILOT_SEQUENCES:
            expected = load_sequence_info(seq).frame_ids
            d = CACHE_DIR / backbone / seq
            row = {"backbone": backbone, "sequence": seq, "cache_dir": str(d)}
            if not (d / ".complete").exists():
                row.update({"status": "fail", "reason": "no .complete flag"})
                rows.append(row)
                continue
            disp = np.load(d / "disparity.npy", mmap_mode="r")
            valid = np.load(d / "valid_mask.npy", mmap_mode="r")
            frame_ids = list(np.load(d / "frame_ids.npy"))
            reasons = []
            if frame_ids != expected:
                reasons.append(f"frame_ids mismatch ({len(frame_ids)} vs {len(expected)} expected)")
            if disp.shape != (len(expected), CACHE_HEIGHT, CACHE_WIDTH):
                reasons.append(f"disp shape {disp.shape}")
            if valid.shape != disp.shape:
                reasons.append(f"valid shape {valid.shape}")
            if disp.dtype != np.float16:
                reasons.append(f"disp dtype {disp.dtype}")
            if valid.dtype != np.uint8:
                reasons.append(f"valid dtype {valid.dtype}")
            row.update({
                "expected_frames": len(expected), "cached_frames": len(frame_ids),
                "status": "pass" if not reasons else "fail", "reason": "; ".join(reasons),
            })
            rows.append(row)
    return rows


def timing_summary() -> list[dict]:
    rows = []
    for backbone in BACKBONE_NAMES:
        for seq in PILOT_SEQUENCES:
            meta_path = CACHE_DIR / backbone / seq / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            rows.append({
                "backbone": backbone, "sequence": seq, "frame_count": meta["frame_count"],
                "runtime_s_total": meta["runtime_s_total"],
                "runtime_ms_per_frame_mean": meta["runtime_ms_per_frame_mean"],
                "runtime_ms_per_frame_median": meta["runtime_ms_per_frame_median"],
                "fps": meta["frame_count"] / meta["runtime_s_total"] if meta["runtime_s_total"] > 0 else None,
            })
    return rows


def storage_summary() -> dict:
    per_job = {}
    total_pilot_bytes = 0
    total_pilot_frames = 0
    for backbone in BACKBONE_NAMES:
        for seq in PILOT_SEQUENCES:
            d = CACHE_DIR / backbone / seq
            if not d.exists():
                continue
            size = dir_size_bytes(d)
            meta = json.loads((d / "metadata.json").read_text())
            per_job[f"{backbone}/{seq}"] = {"bytes": size, "frames": meta["frame_count"]}
            total_pilot_bytes += size
            total_pilot_frames += meta["frame_count"]

    bytes_per_frame = total_pilot_bytes / total_pilot_frames if total_pilot_frames else 0
    full_frame_counts = {}
    with QUALITY_GATE_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["status"] == "pass":
                full_frame_counts[r["sequence_id"]] = int(r["included_count_full"])
    total_full_frames = sum(full_frame_counts.values())
    projected_full_bytes = bytes_per_frame * total_full_frames * len(BACKBONE_NAMES)

    return {
        "pilot_total_bytes": total_pilot_bytes,
        "pilot_total_frames": total_pilot_frames,
        "bytes_per_frame_per_backbone": bytes_per_frame,
        "per_job": per_job,
        "full_17_sequence_total_frames_single_backbone": total_full_frames,
        "projected_full_cache_bytes_5_backbones": projected_full_bytes,
        "projected_full_cache_gb_5_backbones": projected_full_bytes / 1e9,
        "promotion_size_threshold_gb": 15.0,
        "promotion_size_pass": projected_full_bytes / 1e9 < 15.0,
    }


def float16_vs_float32_error() -> list[dict]:
    rows = []
    for backbone in BACKBONE_NAMES:
        for seq in PILOT_SEQUENCES:
            d = CACHE_DIR / backbone / seq
            fp32_path = d / "_fp32_sample.npy"
            if not fp32_path.exists():
                continue
            fp32 = np.load(fp32_path)
            disp16 = np.load(d / "disparity.npy", mmap_mode="r")[: fp32.shape[0]].astype(np.float32)
            err = np.abs(fp32 - disp16)
            rows.append({
                "backbone": backbone, "sequence": seq, "n_sample_frames": fp32.shape[0],
                "mean_abs_error_px": float(err.mean()), "max_abs_error_px": float(err.max()),
                "p99_abs_error_px": float(np.percentile(err, 99)),
            })
    return rows


def random_access_benchmark(n_reads: int = 500) -> dict:
    results = {}
    rng = np.random.default_rng(0)
    for backbone in BACKBONE_NAMES:
        for seq in PILOT_SEQUENCES:
            d = CACHE_DIR / backbone / seq
            if not d.exists():
                continue
            disp = np.load(d / "disparity.npy", mmap_mode="r")
            n = disp.shape[0]
            idx = rng.integers(0, n, size=min(n_reads, n * 5))
            t0 = time.perf_counter()
            acc = 0.0
            for i in idx:
                acc += float(disp[i, 0, 0])  # forces a real page read of that frame's row
            elapsed = time.perf_counter() - t0
            results[f"{backbone}/{seq}"] = {
                "n_reads": len(idx), "total_s": elapsed,
                "ms_per_read": (elapsed / len(idx)) * 1000.0, "reads_per_s": len(idx) / elapsed,
            }
    return results


def colorize_disp(disp: np.ndarray, vmax: float | None = None) -> np.ndarray:
    d = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = vmax or (d.max() if d.max() > 0 else 1.0)
    norm = np.clip(d / vmax * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def contact_sheets(n_frames_per_job: int = 2):
    for backbone in BACKBONE_NAMES:
        for seq in PILOT_SEQUENCES:
            d = CACHE_DIR / backbone / seq
            if not d.exists():
                continue
            info = load_sequence_info(seq)
            disp_cache = np.load(d / "disparity.npy", mmap_mode="r")
            valid_cache = np.load(d / "valid_mask.npy", mmap_mode="r")
            frame_ids = list(np.load(d / "frame_ids.npy"))
            pick_idx = np.linspace(0, len(frame_ids) - 1, n_frames_per_job).astype(int)
            for k in pick_idx:
                frame_id = frame_ids[k]
                left, _right = load_frame_lr(info, frame_id)
                gt_disp, gt_valid = load_frame_gt(info, frame_id)
                gt_small = cv2.resize(gt_disp, (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA) * (
                    CACHE_WIDTH / gt_disp.shape[1]
                )
                gt_valid_small = cv2.resize(gt_valid.astype(np.uint8), (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_NEAREST)
                pred_small = disp_cache[k].astype(np.float32)
                pred_valid_small = valid_cache[k]

                left_small = cv2.resize(cv2.cvtColor(left, cv2.COLOR_RGB2BGR), (CACHE_WIDTH, CACHE_HEIGHT))
                vmax = max(float(gt_small.max()), float(pred_small.max()), 1.0)
                gt_vis = colorize_disp(gt_small, vmax)
                pred_vis = colorize_disp(pred_small, vmax)
                err = np.abs(gt_small.astype(np.float32) - pred_small)
                err_vis = colorize_disp(err, vmax=max(float(err.max()), 1.0))
                valid_vis = cv2.cvtColor(
                    np.clip((pred_valid_small.astype(np.uint16) + gt_valid_small.astype(np.uint16) * 2) * 80, 0, 255).astype(np.uint8),
                    cv2.COLOR_GRAY2BGR,
                )  # 0=neither,80=pred-only,160=gt-only,240=both

                panels = [left_small, gt_vis, pred_vis, err_vis, valid_vis]
                labels = ["RGB", "GT", "pred", "abs err", "validity(pred+2*gt)"]
                strip = np.concatenate(panels, axis=1)
                strip = cv2.copyMakeBorder(strip, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                for i, lab in enumerate(labels):
                    cv2.putText(strip, lab, (i * CACHE_WIDTH + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                out_path = CONTACT_DIR / f"{backbone}_{seq}_{frame_id}.png"
                cv2.imwrite(str(out_path), strip)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    print("cache_integrity...", flush=True)
    integrity_rows = cache_integrity()
    write_csv(OUT_DIR / "cache_integrity.csv", integrity_rows)

    print("timing_summary...", flush=True)
    write_csv(OUT_DIR / "timing_summary.csv", timing_summary())

    print("storage_summary...", flush=True)
    storage = storage_summary()
    (OUT_DIR / "storage_summary.json").write_text(json.dumps(storage, indent=2))

    print("float16_vs_float32_error...", flush=True)
    write_csv(OUT_DIR / "float16_vs_float32_error.csv", float16_vs_float32_error())

    print("random_access_benchmark...", flush=True)
    bench = random_access_benchmark()
    (OUT_DIR / "random_access_benchmark.json").write_text(json.dumps(bench, indent=2))

    print("contact_sheets...", flush=True)
    contact_sheets()

    n_fail = sum(1 for r in integrity_rows if r["status"] == "fail")
    print(json.dumps({
        "integrity_fail_count": n_fail,
        "storage_promotion_pass": storage["promotion_size_pass"],
        "projected_full_cache_gb": storage["projected_full_cache_gb_5_backbones"],
    }, indent=2))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
