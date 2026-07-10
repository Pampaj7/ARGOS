#!/usr/bin/env python3
"""Generate the full-run report suite: cache_integrity_full.csv, timing_summary_full.csv,
storage_summary_full.json, failed_items.csv, run_manifest.json, README.md, and
representative contact sheets. (backbone_metric_summary_cache.csv and
native_resolution_sanity.csv are separate scripts — one reads only the cache, the other
needs GPU inference on the fixed subset.)
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from argos_v2.backbones import BACKBONE_NAMES
from argos_v2.cache_io import validate_written_cache
from argos_v2.paths import ARGOS_ROOT, CACHE_DIR, CACHE_HEIGHT, CACHE_WIDTH, RESULTS_DIR, V2_ROOT
from argos_v2.scared_c_data import load_frame_gt, load_frame_lr, load_sequence_info
from argos_v2.sequences import accepted_sequences, representative_sequences

REPORT_DIR = CACHE_DIR / "reports_full"
CONTACT_DIR = REPORT_DIR / "contact_sheets"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CONTACT_DIR.mkdir(parents=True, exist_ok=True)


def dir_size_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def cache_integrity_full() -> list[dict]:
    rows = []
    for backbone in BACKBONE_NAMES:
        for seq in accepted_sequences():
            info = load_sequence_info(seq)
            d = CACHE_DIR / backbone / seq
            row = {"backbone": backbone, "sequence_id": seq}
            if not d.exists():
                row.update({"integrity_pass": False, "failure_reason": "cache dir missing"})
                rows.append(row)
                continue
            checks = validate_written_cache(backbone, seq, info.frame_ids)
            disp = np.load(d / "disparity.npy", mmap_mode="r") if (d / "disparity.npy").exists() else None
            mask = np.load(d / "valid_mask.npy", mmap_mode="r") if (d / "valid_mask.npy").exists() else None
            row.update({
                "frame_count_expected": len(info.frame_ids),
                "frame_count_actual": checks.get("frame_count_match") and len(info.frame_ids) or None,
                "frame_ids_match": checks.get("frame_ids_exact"),
                "disparity_shape": str(tuple(disp.shape)) if disp is not None else None,
                "disparity_dtype": str(disp.dtype) if disp is not None else None,
                "mask_shape": str(tuple(mask.shape)) if mask is not None else None,
                "mask_dtype": str(mask.dtype) if mask is not None else None,
                "finite_ratio": checks.get("finite_ratio"),
                "prediction_valid_ratio": checks.get("prediction_valid_ratio"),
                "mmap_ok": checks.get("mmap_readable"),
                "completion_flag": (d / ".complete").exists(),
                "integrity_pass": checks.get("passed", False),
                "failure_reason": checks.get("exception", "") if not checks.get("passed") else "",
            })
            rows.append(row)
    return rows


def timing_summary_full() -> list[dict]:
    rows = []
    for backbone in BACKBONE_NAMES:
        for seq in accepted_sequences():
            meta_path = CACHE_DIR / backbone / seq / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            rows.append({
                "backbone": backbone, "sequence_id": seq, "frame_count": meta["frame_count"],
                "total_seconds": meta["total_runtime_s"],
                "mean_seconds_per_frame": meta["avg_runtime_per_frame_s"],
                "median_seconds_per_frame": meta.get("median_runtime_per_frame_s"),
                "p95_seconds_per_frame": meta.get("p95_runtime_per_frame_s"),
                "gpu_assignment": meta.get("device"),
                "peak_gpu_memory_mb": meta.get("peak_gpu_memory_mb"),
            })
    return rows


def storage_summary_full() -> dict:
    per_backbone = {b: 0 for b in BACKBONE_NAMES}
    per_sequence = {}
    total = 0
    for backbone in BACKBONE_NAMES:
        for seq in accepted_sequences():
            d = CACHE_DIR / backbone / seq
            if not d.exists():
                continue
            size = dir_size_bytes(d)
            per_backbone[backbone] += size
            per_sequence.setdefault(seq, 0)
            per_sequence[seq] += size
            total += size
    return {
        "total_bytes": total, "total_gb": total / 1e9,
        "bytes_per_backbone": per_backbone,
        "bytes_per_sequence": per_sequence,
        "bytes_per_frame_avg": total / (16921 * len(BACKBONE_NAMES)) if total else 0,
        "projected_estimate_gb_from_pilot": 6.62,
        "actual_vs_projected_ratio": (total / 1e9) / 6.62 if total else None,
    }


def failed_items() -> list[dict]:
    run_log = RESULTS_DIR / "full_run/run_full.jsonl"
    if not run_log.exists():
        return []
    rows = []
    seen_failed = {}
    for line in run_log.read_text().splitlines():
        r = json.loads(line)
        key = (r["backbone"], r["sequence"])
        if r["returncode"] != 0:
            seen_failed[key] = r
        elif key in seen_failed:
            seen_failed[key] = {**seen_failed[key], "final_status": "ok_after_retry"}
    for key, r in seen_failed.items():
        rows.append({
            "backbone": r["backbone"], "sequence": r["sequence"], "stage": "cache_build",
            "exception": r.get("status", "failed"), "timestamp": r.get("timestamp"),
            "retry_count": r.get("attempt", 1) - 1, "final_status": r.get("final_status", "failed"),
        })
    return rows


def run_manifest() -> dict:
    def git_commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ARGOS_ROOT, text=True).strip()
        except Exception:
            return "unknown"

    checkpoints = {}
    for backbone in BACKBONE_NAMES:
        meta_path = next(iter((CACHE_DIR / backbone).glob("*/metadata.json")), None)
        if meta_path:
            m = json.loads(meta_path.read_text())
            checkpoints[backbone] = {"checkpoint": m.get("checkpoint"), "sha256_partial": m.get("checkpoint_sha256_partial")}

    return {
        "project": "ARGOS v2",
        "script_path": str(Path(__file__).resolve()),
        "orchestrator": str(V2_ROOT / "scripts/run_full.py"),
        "git_commit": git_commit(),
        "sequence_list": accepted_sequences(),
        "backbones": BACKBONE_NAMES,
        "backbone_checkpoints": checkpoints,
        "gpu_scheduling": "LPT (longest-estimated-job-first) shared queue across 2 H100 GPUs, 1 automatic retry on failure",
        "canonical_cache_conventions": {
            "shape": [CACHE_HEIGHT, CACHE_WIDTH], "dtype": "float16",
            "disparity_scale_formula": "d_cache = resize(d_source, (144,180)) * (180.0 / source_width)",
            "disparity_convention": "positive_left_disparity",
        },
        "metric_conventions": "cache-resolution and native-resolution metrics are separate namespaces "
                               "(epe_cache_px vs epe_native_px) and are never converted into each other",
        "gt_resize_fix": "GT resize to cache resolution uses fractional valid-pixel coverage per cell "
                          "(threshold >0.9), not a nearest-neighbor mask over an INTER_AREA-blended value "
                          "array — see argos_v2/metrics.py and scripts/test_gt_resize_regression.py",
        "pilot_reference_results": {
            "S2M2-S_native_epe_px_dataset_3_keyframe_1": 1.07,
            "historical_reference_px": 1.19,
        },
    }


def pick_contact_frame(seq: str, k_fraction: float = 0.5) -> str:
    info = load_sequence_info(seq)
    return info.frame_ids[int(len(info.frame_ids) * k_fraction)]


def colorize(disp: np.ndarray, vmax: float) -> np.ndarray:
    d = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip(d / max(vmax, 1.0) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def contact_sheets():
    reps = representative_sequences()
    for role, seq in reps.items():
        frame_id = pick_contact_frame(seq)
        info = load_sequence_info(seq)
        left, _right = load_frame_lr(info, frame_id)
        gt_disp, gt_valid = load_frame_gt(info, frame_id)
        gt_small = cv2.resize(gt_disp, (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA) * (CACHE_WIDTH / gt_disp.shape[1])
        gt_cov = cv2.resize(gt_valid.astype(np.float32), (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA)
        # Two different masks on purpose: the metric mask (coverage > 0.9, see metrics.py) is
        # intentionally strict to avoid boundary-blend contamination in epe_cache_px numbers.
        # Applying that same 0.9 threshold to a *visualization* leaves ~0.4% of pixels lit
        # (97/25920 on a real frame, measured) — an almost-black panel that looks like a bug.
        # For contact sheets, a relaxed threshold is the right call: readability, not metric
        # rigor, is the point here.
        gt_valid_vis_mask = gt_cov > 0.1
        left_small = cv2.resize(cv2.cvtColor(left, cv2.COLOR_RGB2BGR), (CACHE_WIDTH, CACHE_HEIGHT))

        for backbone in BACKBONE_NAMES:
            d = CACHE_DIR / backbone / seq
            if not (d / ".complete").exists():
                continue
            frame_ids = list(np.load(d / "frame_ids.npy"))
            if frame_id not in frame_ids:
                continue
            k = frame_ids.index(frame_id)
            pred = np.load(d / "disparity.npy", mmap_mode="r")[k].astype(np.float32)
            pred_valid = np.load(d / "valid_mask.npy", mmap_mode="r")[k]

            vmax = max(float(gt_small.max()), float(pred.max()), 1.0)
            gt_vis = colorize(gt_small, vmax)
            pred_vis = colorize(pred, vmax)
            err = np.where(gt_valid_vis_mask, np.abs(gt_small - pred), 0)
            err_vis = colorize(err, max(float(err.max()), 1.0))
            pred_valid_vis = cv2.cvtColor((pred_valid * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            gt_valid_vis = cv2.cvtColor((gt_valid_vis_mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)

            panels = [left_small, gt_vis, pred_vis, err_vis, pred_valid_vis, gt_valid_vis]
            labels = ["RGB", "GT", "pred", "abs err (relaxed mask)", "pred-valid", "gt-valid (relaxed>0.1)"]
            strip = np.concatenate(panels, axis=1)
            strip = cv2.copyMakeBorder(strip, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for i, lab in enumerate(labels):
                cv2.putText(strip, lab, (i * CACHE_WIDTH + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imwrite(str(CONTACT_DIR / f"{role}_{seq}_{backbone}_{frame_id}.png"), strip)
    return reps


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_readme(reps: dict):
    text = f"""# ARGOS-V2 SCARED-C Backbone Cache — Full Run

## Cache structure
```
cache_scaredc_backbones/<backbone>/<sequence_id>/
    disparity.npy    [T,144,180] float16, C-contiguous, uncompressed
    valid_mask.npy   [T,144,180] uint8 (prediction validity only)
    frame_ids.npy    [T]
    metadata.json
```
5 backbones: {", ".join(BACKBONE_NAMES)}. 17 accepted SCARED-C sequences (quality-gate pass).

## Disparity convention
- positive, left-referenced disparity
- units: pixels at cache width 180
- `d_cache = resize(d_source, (144,180)) * (180.0 / source_width)`
- invalid native pixels (non-finite or <=0) are zeroed before resize to avoid blending into
  valid neighbors

## Loading
```python
import numpy as np
disp = np.load("cache_scaredc_backbones/S2M2-S/dataset_3_keyframe_1/disparity.npy", mmap_mode="r")
valid = np.load("cache_scaredc_backbones/S2M2-S/dataset_3_keyframe_1/valid_mask.npy", mmap_mode="r")
frame_ids = np.load("cache_scaredc_backbones/S2M2-S/dataset_3_keyframe_1/frame_ids.npy")
frame_5_disp = disp[5]  # mmap: only this frame is paged in from disk
```

## valid_mask.npy meaning
Prediction validity ONLY (isfinite & >0 at native resolution before resize). This is NOT
GT validity — GT validity lives in SCARED-C's own `gt/*_valid.png` per frame, loaded
separately. Never merge `prediction_valid` and `gt_valid` into one mask irreversibly;
`training_valid = prediction_valid & gt_valid` should be computed by the consumer, not
baked into the cache.

## metadata.json contract
One file per sequence/backbone pair. Key fields: `project`, `backbone`, `checkpoint`,
`checkpoint_sha256_partial`, `sequence_id`, `frame_count`, `source_height/width`,
`model_inference_height/width`, `cache_height/width`, `disparity_dtype`, `disparity_convention`,
`disparity_scale_formula`, `git_commit`, `command_line_args`, `start_time`/`end_time`,
`total_runtime_s`, `avg_runtime_per_frame_s`, `completion_status`, `integrity_validation_status`
(the full 16-point check result). `completion_status` is only `true` after all integrity
checks pass — see `.complete` flag file in the same directory.

## Cache-resolution vs native-resolution metrics — DO NOT CONVERT BETWEEN THEM
Two strictly separate namespaces:
- **Cache-resolution** (`epe_cache_px`, `bad1_cache`, ...): computed at 144x180. Useful for
  cache integrity, relative backbone comparison, and refiner training diagnostics. NEVER
  multiply by an inverse resize factor and report as a native-equivalent number.
- **Native-resolution** (`epe_native_px`, `bad1_native`, ...): computed at full source
  resolution on a fixed 51-frame deterministic subset (`native_validation_subset.json`,
  identical across all 5 backbones), independent of the cache entirely. Use these for
  absolute accuracy sanity and leaderboard comparison.

Confirmed sanity reference: S2M2-S native EPE = 1.07px on `dataset_3_keyframe_1`
(historical reference on the same sequence: ~1.19px) — pipeline confirmed coherent.

## The GT-resize bug (found and fixed during pilot validation)
The first cache-resolution accuracy check resized GT disparity with `INTER_AREA` but used a
**nearest-neighbor** validity mask. Since SCARED-C's invalid pixels are 0-filled at native
resolution, `INTER_AREA` blends those zeros into any cache cell near a valid/invalid
boundary — but the nearest-neighbor mask only samples one source pixel, so contaminated
cells still got marked "valid." This inflated some cache-resolution errors from ~0.5-1px to
~4-10px (and a naive x-native-width rescale made it look like 30-70px "native" error, which
was never a real native-resolution measurement).

**Fix**: a cache cell is only valid if the fractional coverage of valid native pixels inside
its downsampling box exceeds 0.9 (`argos_v2.metrics.resize_gt_to_cache_corrected`). See
`scripts/test_gt_resize_regression.py` for a synthetic regression test that fails under the
old naive resize and passes under the corrected one. **The cache arrays themselves were never
affected** — this was purely an evaluation-script bug; all 5 backbones' predictions are dense
(100% prediction-valid in every job observed), so the same INTER_AREA blending risk on the
*prediction* side never had invalid pixels to blend in the first place.

## Resume
Every job is independently resumable. `run_backbone_cache.py` calls `resume_ok()`, which
re-validates an existing "complete" cache (not just checks the flag file) before skipping —
a corrupted or schema-mismatched cache is silently redone, never silently trusted.
```bash
python3 scripts/run_full.py   # re-run any time; already-valid caches are skipped in ~2s each
```

## How to validate a cache
```bash
python3 -c "
import sys; sys.path.insert(0, 'scripts')
from argos_v2.cache_io import validate_written_cache
from argos_v2.scared_c_data import load_sequence_info
info = load_sequence_info('dataset_3_keyframe_1')
print(validate_written_cache('S2M2-S', 'dataset_3_keyframe_1', info.frame_ids))
"
```

## Representative sequences (contact sheets)
Computed from `quality_gate.csv`, not hand-picked:
- easy (lowest photometric MAE): `{reps['easy']}`
- difficult (highest photometric MAE): `{reps['difficult']}`
- low-coverage (lowest valid_pixel_ratio_mean): `{reps['low_coverage']}`
- boundary-heavy (largest MAE spread, max-median): `{reps['boundary_heavy']}`

## How future training code should load clips
Read `frame_ids.npy` once per sequence/backbone to get temporal order; `disp[i:i+T]` via
mmap gives a contiguous causal clip without loading the whole sequence into RAM. Cross-
reference SCARED-C's own `gt/*_disp.npy` + `gt/*_valid.png` by `frame_id` (not index — GT
and cache share frame IDs but a clip sampler should look them up by ID, not assume aligned
indexing across different backbones' caches, since a resumed/regenerated cache always
preserves the source frame order but different backbones are built as independent jobs).
"""
    (REPORT_DIR / "README.md").write_text(text)


def main() -> int:
    print("cache_integrity_full...", flush=True)
    integrity = cache_integrity_full()
    write_csv(REPORT_DIR / "cache_integrity_full.csv", integrity)

    print("timing_summary_full...", flush=True)
    write_csv(REPORT_DIR / "timing_summary_full.csv", timing_summary_full())

    print("storage_summary_full...", flush=True)
    storage = storage_summary_full()
    (REPORT_DIR / "storage_summary_full.json").write_text(json.dumps(storage, indent=2))

    print("failed_items...", flush=True)
    write_csv(REPORT_DIR / "failed_items.csv", failed_items())

    print("run_manifest...", flush=True)
    (REPORT_DIR / "run_manifest.json").write_text(json.dumps(run_manifest(), indent=2))

    print("contact_sheets...", flush=True)
    reps = contact_sheets()

    print("README...", flush=True)
    write_readme(reps)

    n_fail = sum(1 for r in integrity if not r.get("integrity_pass"))
    print(json.dumps({"total_pairs": len(integrity), "integrity_fail_count": n_fail,
                       "storage_gb": storage["total_gb"]}, indent=2))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
