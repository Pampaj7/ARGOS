import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from eval_scared_fast_foundationstereo_onnx import infer, make_session
from eval_scared_s2m2 import disp_vis, load_gt, read_rgb


def depth_vis(depth_mm, mask=None, max_val=None):
    valid = np.isfinite(depth_mm)
    if mask is not None:
        valid &= mask
    if max_val is None:
        max_val = np.percentile(depth_mm[valid], 99) if valid.any() else 1.0
    x = np.zeros_like(depth_mm, dtype=np.float32)
    x[valid] = depth_mm[valid]
    x = np.clip(x, 0, max(max_val, 1e-6))
    x = (x / max(max_val, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)[..., ::-1]


def mask_vis(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = (255, 255, 255)
    return out


def label_panel(img, label):
    canvas = np.full((img.shape[0] + 24, img.shape[1], 3), 255, dtype=np.uint8)
    canvas[24:] = img
    cv2.putText(canvas, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def make_frame_row(label, left, gt_depth, pred_disp, pred_depth, abs_error, near_zero, valid_mask):
    depth_max = np.percentile(gt_depth[valid_mask], 99) if valid_mask.any() else 1.0
    disp_max = np.percentile(pred_disp[valid_mask], 99) if valid_mask.any() else 1.0
    finite_error = np.isfinite(abs_error) & valid_mask
    err_max = min(5000.0, np.percentile(abs_error[finite_error], 99.5)) if finite_error.any() else 1.0
    pred_depth_cap = min(5000.0, np.percentile(pred_depth[np.isfinite(pred_depth) & valid_mask], 99.5))

    panels = [
        label_panel(left, f"{label} | left"),
        label_panel(depth_vis(gt_depth, valid_mask, depth_max), "GT depth"),
        label_panel(disp_vis(pred_disp, disp_max), "pred disp"),
        label_panel(depth_vis(pred_depth, valid_mask, pred_depth_cap), "pred depth"),
        label_panel(depth_vis(abs_error, valid_mask, err_max), "abs depth err"),
        label_panel(mask_vis(near_zero & valid_mask), "pred disp < 0.1"),
    ]
    thumbs = [cv2.resize(p, (240, 216), interpolation=cv2.INTER_AREA) for p in panels]
    return np.concatenate(thumbs, axis=1)


def write_csv(path, rows, keys):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared_root", default="stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes")
    parser.add_argument(
        "--model_file",
        default="stereo/Fast-FoundationStereo/weights/onnx/20_30_48/320x736/20_30_48_iters_4_res_320x736.onnx",
    )
    parser.add_argument("--out_dir", default="results/scared_fast_foundationstereo_onnx_outlier_audit")
    parser.add_argument("--montage_frames", type=int, default=10)
    args = parser.parse_args()

    model_file = Path(args.model_file)
    cfg = yaml.safe_load(model_file.with_suffix(".yaml").read_text())
    target_h, target_w = cfg["image_size"]
    session = make_session(model_file)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    rows = []
    top_pixel_candidates = []
    frame_payloads = {}
    root = Path(args.scared_root)
    for exp_dir in sorted(root.glob("dataset_*")):
        ref_dir = exp_dir / "Reference_SCARED"
        for left_path in sorted((exp_dir / "Left_rectified").glob("*.png")):
            stem = left_path.stem
            right_path = exp_dir / "Right_rectified" / f"{stem}.png"
            calib_path = exp_dir / "Rectified_calibration" / f"{stem}.json"
            if not all(p.exists() for p in [right_path, calib_path]):
                continue

            label = f"{exp_dir.name}/{stem}"
            left = read_rgb(left_path)
            right = read_rgb(right_path)
            pred_disp = infer(session, left, right, target_h, target_w)

            gt_disp, gt_depth, gt_mask, gt_source = load_gt(ref_dir, stem)
            calib = json.loads(calib_path.read_text())
            p1 = np.array(calib["P1"]["data"], dtype=np.float32).reshape(3, 4)
            p2 = np.array(calib["P2"]["data"], dtype=np.float32).reshape(3, 4)
            focal = float(p1[0, 0])
            baseline_mm = float(abs(p2[0, 3] / p2[0, 0]))
            pred_depth = focal * baseline_mm / np.maximum(pred_disp, 1e-6)
            valid = gt_mask & np.isfinite(pred_depth)
            abs_error = np.abs(pred_depth - gt_depth)
            valid_error = abs_error[valid]
            valid_pred_depth = pred_depth[valid]
            valid_pred_disp = pred_disp[valid]
            near_zero_01 = valid & (pred_disp < 0.1)
            near_zero_05 = valid & (pred_disp < 0.5)

            safe_name = f"{exp_dir.name}_{stem}"
            np.save(pred_dir / f"{safe_name}_pred_disparity_float32.npy", pred_disp.astype(np.float32))
            np.save(pred_dir / f"{safe_name}_pred_depth_float32.npy", pred_depth.astype(np.float32))

            near_zero_01_error_sum = float(abs_error[near_zero_01].sum()) if near_zero_01.any() else 0.0
            total_error_sum = float(valid_error.sum()) if valid_error.size else 0.0
            top_count = min(200, int(valid.sum()))
            if top_count:
                valid_flat = valid.ravel()
                error_flat = abs_error.ravel()
                valid_indices = np.flatnonzero(valid_flat)
                valid_errors = error_flat[valid_indices]
                local = np.argpartition(valid_errors, -top_count)[-top_count:]
                for flat_idx in valid_indices[local]:
                    y, x = np.unravel_index(int(flat_idx), valid.shape)
                    top_pixel_candidates.append(
                        {
                            "dataset": exp_dir.name,
                            "frame": stem,
                            "label": label,
                            "y": int(y),
                            "x": int(x),
                            "pred_disp_px": float(pred_disp[y, x]),
                            "pred_depth_mm": float(pred_depth[y, x]),
                            "gt_depth_mm": float(gt_depth[y, x]),
                            "abs_depth_error_mm": float(abs_error[y, x]),
                            "near_zero_disp_lt_0_1": bool(pred_disp[y, x] < 0.1),
                            "near_zero_disp_lt_0_5": bool(pred_disp[y, x] < 0.5),
                        }
                    )
            row = {
                "dataset": exp_dir.name,
                "frame": stem,
                "label": label,
                "gt_source": gt_source,
                "valid_px": int(valid.sum()),
                "valid_pixel_ratio": float(valid.mean()),
                "raw_depth_mae_mm": float(valid_error.mean()),
                "raw_depth_rmse_mm": float(np.sqrt((valid_error**2).mean())),
                "raw_depth_median_abs_error_mm": float(np.median(valid_error)),
                "max_pred_depth_mm": float(valid_pred_depth.max()),
                "p99_pred_depth_mm": float(np.percentile(valid_pred_depth, 99)),
                "p999_pred_depth_mm": float(np.percentile(valid_pred_depth, 99.9)),
                "min_pred_disp_px": float(valid_pred_disp.min()),
                "p01_pred_disp_px": float(np.percentile(valid_pred_disp, 1)),
                "near_zero_disp_count_lt_0_1": int(near_zero_01.sum()),
                "near_zero_disp_count_lt_0_5": int(near_zero_05.sum()),
                "near_zero_disp_ratio_lt_0_1": float(near_zero_01.sum() / max(valid.sum(), 1)),
                "near_zero_disp_ratio_lt_0_5": float(near_zero_05.sum() / max(valid.sum(), 1)),
                "near_zero_lt_0_1_error_sum_mm": near_zero_01_error_sum,
                "near_zero_lt_0_1_error_fraction": float(near_zero_01_error_sum / total_error_sum) if total_error_sum else 0.0,
            }
            rows.append(row)
            frame_payloads[label] = (left, gt_depth, pred_disp, pred_depth, abs_error, near_zero_01, valid)
            print(label, flush=True)

    if not rows:
        raise RuntimeError("No SCARED converted frames found")

    keys = list(rows[0].keys())
    write_csv(out_dir / "per_frame_outlier_audit.csv", rows, keys)

    by_mae = sorted(rows, key=lambda r: r["raw_depth_mae_mm"], reverse=True)[:10]
    by_max_depth = sorted(rows, key=lambda r: r["max_pred_depth_mm"], reverse=True)[:10]
    write_csv(out_dir / "top10_worst_frames_by_raw_depth_mae.csv", by_mae, keys)
    write_csv(out_dir / "top10_worst_frames_by_max_pred_depth.csv", by_max_depth, keys)
    top_pixels = sorted(top_pixel_candidates, key=lambda r: r["abs_depth_error_mm"], reverse=True)[:500]
    write_csv(out_dir / "top_depth_error_pixels.csv", top_pixels, list(top_pixels[0].keys()))

    selected = []
    seen = set()
    for row in by_mae + by_max_depth:
        if row["label"] not in seen:
            selected.append(row)
            seen.add(row["label"])
        if len(selected) >= args.montage_frames:
            break

    montage_rows = []
    for row in selected:
        montage_rows.append(make_frame_row(row["label"], *frame_payloads[row["label"]]))
    cv2.imwrite(str(out_dir / "montage_outlier_frames.png"), np.concatenate(montage_rows, axis=0)[..., ::-1])

    summary = {
        "frames": len(rows),
        "onnxruntime_providers": session.get_providers(),
        "image_size": [target_h, target_w],
        "top_raw_depth_mae": by_mae[:3],
        "top_max_pred_depth": by_max_depth[:3],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
