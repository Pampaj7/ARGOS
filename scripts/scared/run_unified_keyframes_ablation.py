#!/usr/bin/env python3
"""Redo the "which stereo model is best on SCARED" ablation on all 45 fixed-rectification
strong keyframes (dataset_1,2,3,4,5,6,7,8,9), replacing the old `unified_keyframes/`
report which used a buggy cv2.remap-based GT rectification (see DATASET_CARD.md /
scared-strong-keyframes-reorg memory) and only covered dataset_8 (5 keyframes).

Every model here scores against the exact same GT (collect_samples ->
scatter_min_depth, R1-rotation + z-buffer projection) so results are directly comparable.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR, RESULTS_DIR, EXTERNAL_DIR

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_s2m2_size_tradeoff import collect_samples  # noqa: E402
from eval_metrics import failure_aware_metrics, summarize_rows  # noqa: E402
from eval_scared_external_native import (  # noqa: E402
    eval_crestereo, eval_raft, eval_defom, eval_monster, eval_stereoanywhere, image_to_tensor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "temporal_refinement/data_prep"))
from predict_s2m2_long_sequences import build_model as build_s2m2, infer as infer_s2m2  # noqa: E402

SCARED_ROOT = DATASET_DIR / "SCARED/curated/geometric_gt/strong_keyframes"
OUT = RESULTS_DIR / "01_frame_stereo/SCARED/unified_keyframes"
FAST_FS_ONNX = EXTERNAL_DIR / "frame_stereo_repos/Fast-FoundationStereo/weights/onnx/20_30_48/320x736/20_30_48_iters_4_res_320x736.onnx"


def metric_row(pred: np.ndarray, sample: dict) -> dict:
    pred = np.clip(pred.astype(np.float32), 0, None)
    pred_depth = sample["fx"] * sample["baseline_mm"] / np.maximum(pred, 1e-6)
    raw_mask = (
        sample["valid"] & np.isfinite(pred) & np.isfinite(pred_depth)
        & (sample["gt_disp"] > 0) & (sample["gt_depth"] > 0)
    )
    return failure_aware_metrics(pred, pred_depth, sample["gt_disp"], sample["gt_depth"], sample["valid"], raw_mask)


def run_model(name: str, method: str, checkpoint: str, predict, samples: list[dict], out_dir: Path) -> None:
    rows = []
    for s in samples:
        t0 = time.perf_counter()
        pred, runtime_ms = predict(s["left"], s["right"])
        if not isinstance(runtime_ms, float):
            runtime_ms = (time.perf_counter() - t0) * 1000.0
        row = {"model": method, "checkpoint": checkpoint, "frame": s["frame"], "runtime_ms": runtime_ms}
        row.update(metric_row(pred, s))
        rows.append(row)
        print(f"  {name}/{s['frame']}: mae={row['mae_px']:.3f} depth_mae_mm={row['depth_mae_mm']:.3f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summarize_rows(rows), indent=2))


def sgbm_predict(left, right):
    left_g = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right_g = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=256, blockSize=5,
        P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
    )
    t0 = time.perf_counter()
    disp = matcher.compute(left_g, right_g).astype(np.float32) / 16.0
    ms = (time.perf_counter() - t0) * 1000.0
    return np.clip(disp, 0, None), ms


def make_fast_foundationstereo_predict(device):
    import onnxruntime as ort
    import yaml

    providers = (["CUDAExecutionProvider"] if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers() else []) + ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(FAST_FS_ONNX), providers=providers)
    cfg = yaml.safe_load(FAST_FS_ONNX.with_suffix(".yaml").read_text())
    target_h, target_w = cfg["image_size"]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict(left, right):
        orig_h, orig_w = left.shape[:2]
        sx = target_w / float(orig_w)
        left_res = cv2.resize(left, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        right_res = cv2.resize(right, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        left_t = (((left_res.astype(np.float32) / 255.0) - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        right_t = (((right_res.astype(np.float32) / 255.0) - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        t0 = time.perf_counter()
        outputs = session.run(None, {"left_image": left_t, "right_image": right_t})
        ms = (time.perf_counter() - t0) * 1000.0
        pred = outputs[0].reshape(target_h, target_w).astype(np.float32).clip(0, None)
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / sx
        return pred.astype(np.float32), ms

    return predict


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = SimpleNamespace()
    print(f"collecting samples from {SCARED_ROOT} ...", flush=True)
    samples = collect_samples(SCARED_ROOT)
    print(f"{len(samples)} keyframes loaded", flush=True)

    root_cwd = os.getcwd()
    external_builders = {
        "CREStereo_native": eval_crestereo,
        "RAFT-Stereo_Middlebury_native": eval_raft,
        "DEFOM-Stereo_ViT-L_ETH3D_native": eval_defom,
        "MonSter++_MixAll_native": lambda s, a, d: eval_monster(s, a, d, realtime=False),
        "RT-MonSter++_ZeroShot_native": lambda s, a, d: eval_monster(s, a, d, realtime=True),
        "StereoAnywhere_native": eval_stereoanywhere,
    }
    for name, builder in external_builders.items():
        print(f"=== {name} ===", flush=True)
        try:
            method, checkpoint, _res, predict = builder(samples, args, device)
            run_model(name, method, checkpoint, predict, samples, OUT / name)
        except Exception as e:
            print(f"SKIPPED {name}: {type(e).__name__}: {e}", flush=True)
        finally:
            os.chdir(root_cwd)

    print("=== S2M2-S / S2M2-XL ===", flush=True)
    for variant, name in [("S", "S2M2-S"), ("XL", "S2M2-XL")]:
        model = build_s2m2(device, variant)

        width = 512  # matched across variants: SCARED is native 1024x1280, both S/XL score
        # best at low resize width on this dataset (verified via width ablation:
        # S/XL MAE at 512/736/1024/full-res = 3.08/4.05/5.23/5.87 and 3.86/4.90/4.72/6.95px);
        # using the same width for both keeps the S-vs-XL comparison apples-to-apples.

        def predict(left, right, model=model, width=width):
            pred, ms, _scale = infer_s2m2(model, left, right, device, width)
            return pred, ms

        run_model(name, f"S2M2-{variant}", f"CH{'128NTR1' if variant == 'S' else '384NTR3'}.pth", predict, samples, OUT / name)

    print("=== SGBM ===", flush=True)
    run_model("SGBM", "OpenCV SGBM", "n/a", sgbm_predict, samples, OUT / "SGBM")

    print("=== Fast-FoundationStereo ONNX ===", flush=True)
    try:
        predict = make_fast_foundationstereo_predict(device)
        run_model("Fast-FoundationStereo_ONNX", "Fast-FoundationStereo 20_30_48 iters=4 320x736", "onnx", predict, samples, OUT / "Fast-FoundationStereo_ONNX")
    except Exception as e:
        print(f"SKIPPED Fast-FoundationStereo_ONNX: {type(e).__name__}: {e}", flush=True)

    # cross-model leaderboard
    leaderboard = []
    for model_dir in sorted(OUT.iterdir()):
        summary_path = model_dir / "summary.json"
        if summary_path.exists():
            row = {"model": model_dir.name}
            row.update(json.loads(summary_path.read_text()))
            leaderboard.append(row)
    if leaderboard:
        cols = list(leaderboard[0].keys())
        with (OUT / "leaderboard.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(leaderboard)
    (OUT / "README.md").write_text(
        f"# Unified SCARED keyframe ablation (fixed rectification, {len(samples)} keyframes)\n\n"
        "GT: scatter_min_depth (R1-rotation + z-buffer projection), replaces the retired "
        "cv2.remap-based rectification used in the previous version of this report.\n\n"
        "See leaderboard.csv for the cross-model summary.\n"
    )
    print(json.dumps({"n_keyframes": len(samples), "n_models": len(leaderboard)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
