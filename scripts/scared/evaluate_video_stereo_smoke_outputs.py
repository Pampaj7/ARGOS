#!/usr/bin/env python3
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ARGOS = Path("/home/pampaj/Desktop/ARGOS")
OUT = ARGOS / "results/video_stereo_repos"
TEST = OUT / "test_sequence"
S2M2_SRC = Path("/home/pampaj/Desktop/stereo/s2m2/src")
S2M2_WEIGHTS = Path("/home/pampaj/Desktop/stereo/s2m2/weights/pretrain_weights")
SCARED_ROOT = ARGOS / "dataset/scared_keyframes_gt_dataset8/dataset_8"


def load_test_sequence():
    metadata = json.loads((TEST / "metadata.json").read_text())
    frames = []
    for frame in metadata["frames"]:
        left = cv2.cvtColor(cv2.imread(str(TEST / frame["left"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(str(TEST / frame["right"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gt_disp = np.load(TEST / frame["gt_disparity"]).astype(np.float32)
        gt_depth = np.load(TEST / frame["gt_depth"]).astype(np.float32)
        valid = cv2.imread(str(TEST / frame["valid_mask"]), cv2.IMREAD_GRAYSCALE) > 0
        frames.append({**frame, "left_image": left, "right_image": right, "gt_disp_array": gt_disp, "gt_depth_array": gt_depth, "valid_array": valid})
    return frames


def metrics(name, pred_stack, frames, runtime_ms=None, peak_mb=None):
    rows = []
    temporal_abs = []
    temporal_error_var = []
    errors = []
    prev_pred = None
    prev_err = None
    for idx, (pred, frame) in enumerate(zip(pred_stack, frames)):
        gt_disp = frame["gt_disp_array"]
        gt_depth = frame["gt_depth_array"]
        valid = frame["valid_array"] & (gt_disp > 0) & (gt_depth > 0) & np.isfinite(pred)
        pred_depth = frame["fx"] * frame["baseline_mm"] / np.maximum(pred, 1e-6)
        disp_err = np.abs(pred[valid] - gt_disp[valid])
        depth_err = np.abs(pred_depth[valid] - gt_depth[valid])
        err_map = np.full_like(gt_disp, np.nan, dtype=np.float32)
        err_map[valid] = np.abs(pred[valid] - gt_disp[valid])
        errors.append(err_map)
        if prev_pred is not None:
            both = valid & np.isfinite(prev_pred)
            temporal_abs.append(float(np.mean(np.abs(pred[both] - prev_pred[both]))))
            both_err = both & np.isfinite(prev_err)
            temporal_error_var.append(float(np.mean(np.abs(err_map[both_err] - prev_err[both_err]))))
        prev_pred = pred
        prev_err = err_map
        rows.append(
            {
                "model_name": name,
                "frame": frame["index"],
                "valid_disp_mae": float(disp_err.mean()),
                "valid_disp_rmse": float(np.sqrt((disp_err**2).mean())),
                "bad_1px": float((disp_err > 1).mean() * 100),
                "bad_2px": float((disp_err > 2).mean() * 100),
                "bad_3px": float((disp_err > 3).mean() * 100),
                "valid_depth_mae": float(depth_err.mean()),
                "valid_depth_median": float(np.median(depth_err)),
                "valid_depth_rmse": float(np.sqrt((depth_err**2).mean())),
                "bad_1mm": float((depth_err > 1).mean() * 100),
                "bad_2mm": float((depth_err > 2).mean() * 100),
                "bad_5mm": float((depth_err > 5).mean() * 100),
                "pred_disp_le_0_1_ratio": float((frame["valid_array"] & (pred <= 0.1)).sum() / max(int(frame["valid_array"].sum()), 1)),
                "pred_disp_le_0_5_ratio": float((frame["valid_array"] & (pred <= 0.5)).sum() / max(int(frame["valid_array"].sum()), 1)),
                "valid_pixel_ratio": float(valid.mean()),
            }
        )
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "frame"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update(
        {
            "model_name": name,
            "frames": len(rows),
            "avg_runtime_ms": runtime_ms,
            "peak_gpu_memory_mb": peak_mb,
            "mean_abs_consecutive_pred_disp_diff": float(np.mean(temporal_abs)) if temporal_abs else None,
            "per_pixel_temporal_std": float(np.nanmean(np.nanstd(np.stack(pred_stack), axis=0))),
            "temporal_error_variation": float(np.mean(temporal_error_var)) if temporal_error_var else None,
        }
    )
    return summary, rows


def load_stereoanyvideo(frames):
    pred = np.load(OUT / "StereoAnyVideo/outputs/disparity.npy").astype(np.float32)
    if pred.ndim == 4:
        pred = pred[:, 0]
    orig_h, orig_w = frames[0]["gt_disp_array"].shape
    in_h, in_w = pred.shape[-2:]
    scale_x = in_w / float(orig_w)
    out = []
    for item in pred:
        out.append(cv2.resize(item, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale_x)
    summary = json.loads((OUT / "StereoAnyVideo/outputs/summary.json").read_text())
    return out, None, None, summary


def build_s2m2(model_type, refine_iter=3):
    sys.path.insert(0, str(S2M2_SRC))
    from s2m2.core.model.s2m2 import S2M2

    cfg = {"S": (128, 1), "L": (256, 3), "XL": (384, 3)}[model_type]
    model = S2M2(feature_channels=cfg[0], dim_expansion=1, num_transformer=cfg[1], use_positivity=True, refine_iter=refine_iter)
    ckpt = torch.load(S2M2_WEIGHTS / f"CH{cfg[0]}NTR{cfg[1]}.pth", map_location="cpu", weights_only=True)
    model.my_load_state_dict(ckpt["state_dict"])
    return model.cuda().eval()


def infer_s2m2(model, left, right, width):
    sys.path.insert(0, str(S2M2_SRC))
    from s2m2.core.utils.image_utils import image_crop, image_pad

    orig_h, orig_w = left.shape[:2]
    if width == 0:
        left_in, right_in, scale_x = left, right, 1.0
    else:
        scale_x = width / float(orig_w)
        new_h = int(round(orig_h * scale_x))
        left_in = cv2.resize(left, (width, new_h), interpolation=cv2.INTER_LINEAR)
        right_in = cv2.resize(right, (width, new_h), interpolation=cv2.INTER_LINEAR)
    left_t = torch.from_numpy(left_in).permute(2, 0, 1).unsqueeze(0).float().cuda()
    right_t = torch.from_numpy(right_in).permute(2, 0, 1).unsqueeze(0).float().cuda()
    h, w = left_t.shape[-2:]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        pred, _occ, _conf = model(image_pad(left_t, 32), image_pad(right_t, 32))
    torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - t0) * 1000
    pred = image_crop(pred, (h, w)).squeeze().float().cpu().numpy()
    if scale_x != 1:
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale_x
    return np.clip(pred.astype(np.float32), 0, None), runtime_ms


def run_s2m2(name, model_type, width, frames):
    out_dir = OUT / "frame_baselines" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_s2m2(model_type)
    preds = []
    runtimes = []
    peak = 0.0
    for idx, frame in enumerate(frames):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pred, runtime_ms = infer_s2m2(model, frame["left_image"], frame["right_image"], width)
        peak = max(peak, torch.cuda.max_memory_allocated() / (1024**2))
        np.save(out_dir / f"{idx:06d}.npy", pred)
        preds.append(pred)
        runtimes.append(runtime_ms)
    del model
    torch.cuda.empty_cache()
    return preds, float(np.mean(runtimes)), peak


def main():
    frames = load_test_sequence()
    all_summaries = []
    all_rows = []

    sav_preds, _sav_rt, _sav_mem, sav_raw_summary = load_stereoanyvideo(frames)
    summary, rows = metrics("StereoAnyVideo@384x640", sav_preds, frames, None, None)
    summary["smoke_summary"] = sav_raw_summary
    all_summaries.append(summary)
    all_rows.extend(rows)

    for name, model_type, width in [("S2M2-L@736", "L", 736), ("S2M2-L@full", "L", 0), ("S2M2-S@512", "S", 512)]:
        preds, runtime_ms, peak_mb = run_s2m2(name, model_type, width, frames)
        summary, rows = metrics(name, preds, frames, runtime_ms, peak_mb)
        all_summaries.append(summary)
        all_rows.extend(rows)

    with (OUT / "smoke_metrics.json").open("w") as f:
        json.dump({"summary": all_summaries, "frames": all_rows}, f, indent=2)
    with (OUT / "smoke_metrics.csv").open("w", newline="") as f:
        keys = [k for k in all_summaries[0].keys() if k != "smoke_summary"] + ["smoke_summary"]
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_summaries)
    print(json.dumps({"summary": all_summaries}, indent=2))


if __name__ == "__main__":
    main()

