import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_scared_s2m2 import build_model, disp_vis, infer, load_pretrained, read_rgb


def depth_from_disp(disp, calib_path):
    calib = json.loads(Path(calib_path).read_text())
    p1 = np.array(calib["P1"]["data"], dtype=np.float32).reshape(3, 4)
    p2 = np.array(calib["P2"]["data"], dtype=np.float32).reshape(3, 4)
    f = p1[0, 0]
    baseline = abs(p2[0, 3] / p2[0, 0])
    return f * baseline / np.maximum(disp, 1e-6)


def color_depth(depth, mask, max_depth):
    x = np.zeros_like(depth, dtype=np.float32)
    x[mask] = depth[mask]
    x = np.clip(x, 0, max(max_depth, 1e-6))
    x = (x / max(max_depth, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)[..., ::-1]


def color_error(err, max_err=10.0):
    x = np.clip(err, 0, max_err)
    x = (x / max(max_err, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)[..., ::-1]


def main():
    root = Path("stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes")
    out_dir = Path("results/scared_s2m2_XL_resize_vs_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model("XL", 3, device)
    load_pretrained(model, "stereo/s2m2/weights/pretrain_weights", "XL")

    bins = np.array([0, 20, 30, 40, 50, 60, 80, 100, 130, 170, 260], dtype=np.float32)
    resize_bin_errors = [[] for _ in range(len(bins) - 1)]
    full_bin_errors = [[] for _ in range(len(bins) - 1)]
    rows = []
    montage_rows = []

    for exp_dir in sorted(root.glob("dataset_*")):
        ref_dir = exp_dir / "Reference_SCARED"
        for left_path in sorted((exp_dir / "Left_rectified").glob("*.png")):
            stem = left_path.stem
            right_path = exp_dir / "Right_rectified" / f"{stem}.png"
            gt_depth_path = ref_dir / "DepthL" / f"{stem}.png"
            calib_path = exp_dir / "Rectified_calibration" / f"{stem}.json"
            left = read_rgb(left_path)
            right = read_rgb(right_path)
            gt_depth = cv2.imread(str(gt_depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            mask = gt_depth > 0

            pred_resize_disp = infer(model, left, right, device, 1024)
            pred_full_disp = infer(model, left, right, device, 0)
            pred_resize_depth = depth_from_disp(pred_resize_disp, calib_path)
            pred_full_depth = depth_from_disp(pred_full_disp, calib_path)
            err_resize = np.abs(pred_resize_depth - gt_depth)
            err_full = np.abs(pred_full_depth - gt_depth)

            for i in range(len(bins) - 1):
                bm = mask & (gt_depth >= bins[i]) & (gt_depth < bins[i + 1])
                if bm.any():
                    resize_bin_errors[i].append(err_resize[bm])
                    full_bin_errors[i].append(err_full[bm])

            rows.append(
                {
                    "dataset": exp_dir.name,
                    "frame": stem,
                    "valid_pixel_ratio": float(mask.mean()),
                    "resize_depth_mae_mm": float(err_resize[mask].mean()),
                    "resize_depth_median_abs_error_mm": float(np.median(err_resize[mask])),
                    "resize_depth_bad1mm_pct": float((err_resize[mask] > 1).mean() * 100),
                    "resize_depth_bad2mm_pct": float((err_resize[mask] > 2).mean() * 100),
                    "resize_depth_bad5mm_pct": float((err_resize[mask] > 5).mean() * 100),
                    "full_depth_mae_mm": float(err_full[mask].mean()),
                    "full_depth_median_abs_error_mm": float(np.median(err_full[mask])),
                    "full_depth_bad1mm_pct": float((err_full[mask] > 1).mean() * 100),
                    "full_depth_bad2mm_pct": float((err_full[mask] > 2).mean() * 100),
                    "full_depth_bad5mm_pct": float((err_full[mask] > 5).mean() * 100),
                }
            )

            if len(montage_rows) < 18:
                max_depth = np.percentile(gt_depth[mask], 99)
                panel = np.concatenate(
                    [
                        left,
                        color_depth(gt_depth, mask, max_depth),
                        color_depth(pred_resize_depth, mask, max_depth),
                        color_depth(pred_full_depth, mask, max_depth),
                        color_error(err_resize, 10.0),
                        color_error(err_full, 10.0),
                    ],
                    axis=1,
                )
                thumb = cv2.resize(panel, (1800, 300), interpolation=cv2.INTER_AREA)
                canvas = np.full((324, 1800, 3), 255, dtype=np.uint8)
                cv2.putText(canvas, f"{exp_dir.name}/{stem}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
                canvas[24:] = thumb
                montage_rows.append(canvas)

    with open(out_dir / "per_frame_resize_vs_full.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bin_rows = []
    for i in range(len(bins) - 1):
        r = np.concatenate(resize_bin_errors[i]) if resize_bin_errors[i] else np.array([])
        f = np.concatenate(full_bin_errors[i]) if full_bin_errors[i] else np.array([])
        bin_rows.append(
            {
                "depth_bin_start_mm": float(bins[i]),
                "depth_bin_end_mm": float(bins[i + 1]),
                "resize_mae_mm": float(r.mean()) if r.size else None,
                "resize_median_mm": float(np.median(r)) if r.size else None,
                "full_mae_mm": float(f.mean()) if f.size else None,
                "full_median_mm": float(np.median(f)) if f.size else None,
                "pixels": int(r.size),
            }
        )
    with open(out_dir / "error_vs_depth_bins.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bin_rows)

    centers = (bins[:-1] + bins[1:]) / 2
    plt.figure(figsize=(8, 5))
    plt.plot(centers, [r["resize_mae_mm"] for r in bin_rows], marker="o", label="resize 1024 MAE")
    plt.plot(centers, [r["full_mae_mm"] for r in bin_rows], marker="o", label="full-res MAE")
    plt.plot(centers, [r["resize_median_mm"] for r in bin_rows], marker="x", linestyle="--", label="resize 1024 median")
    plt.plot(centers, [r["full_median_mm"] for r in bin_rows], marker="x", linestyle="--", label="full-res median")
    plt.xlabel("GT depth bin center (mm)")
    plt.ylabel("Absolute depth error (mm)")
    plt.title("S2M2-XL SCARED keyframes: error vs depth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_depth.png", dpi=180)
    plt.close()

    cv2.imwrite(str(out_dir / "montage_left_gt_resize_full_errors.png"), np.concatenate(montage_rows, axis=0)[..., ::-1])
    print(out_dir)


if __name__ == "__main__":
    main()
