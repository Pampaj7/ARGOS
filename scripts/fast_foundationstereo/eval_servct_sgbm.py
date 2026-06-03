import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")
from Utils import vis_disparity


def metrics(pred, gt, mask):
    err = np.abs(pred[mask] - gt[mask])
    return {
        "valid_px": int(mask.sum()),
        "mae_px": float(err.mean()),
        "rmse_px": float(np.sqrt((err ** 2).mean())),
        "bad1_pct": float((err > 1.0).mean() * 100.0),
        "bad2_pct": float((err > 2.0).mean() * 100.0),
        "bad5_pct": float((err > 5.0).mean() * 100.0),
    }


def depth_metrics(pred_mm, gt_mm, mask):
    err = np.abs(pred_mm[mask] - gt_mm[mask])
    return {
        "depth_mae_mm": float(err.mean()),
        "depth_rmse_mm": float(np.sqrt((err ** 2).mean())),
        "depth_bad1mm_pct": float((err > 1.0).mean() * 100.0),
        "depth_bad2mm_pct": float((err > 2.0).mean() * 100.0),
        "depth_bad5mm_pct": float((err > 5.0).mean() * 100.0),
    }


def make_sgbm(max_disp, block_size):
    max_disp = int(np.ceil(max_disp / 16.0) * 16)
    channels = 1
    p1 = 8 * channels * block_size * block_size
    p2 = 32 * channels * block_size * block_size
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max_disp,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=2,
        uniquenessRatio=4,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--servct_root", default="data/surgical_stereo/servct/SERV-CT")
    parser.add_argument("--reference", choices=["Reference_CT", "Reference_RGB"], default="Reference_CT")
    parser.add_argument("--out_dir", default="output_servct_eval_sgbm")
    parser.add_argument("--block_size", type=int, default=3)
    parser.add_argument("--max_disp", type=int, default=192)
    args = parser.parse_args()

    root = Path(args.servct_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matcher = make_sgbm(args.max_disp, args.block_size)

    rows = []
    montage_rows = []
    for exp in ["Experiment_1", "Experiment_2"]:
        exp_dir = root / exp
        ref_dir = exp_dir / args.reference
        if not ref_dir.exists():
            continue
        for left_path in sorted((exp_dir / "Left_rectified").glob("*.png")):
            stem = left_path.stem
            right_path = exp_dir / "Right_rectified" / f"{stem}.png"
            gt_disp_path = ref_dir / "Disparity" / f"{stem}.png"
            gt_depth_path = ref_dir / "DepthL" / f"{stem}.png"
            calib_path = exp_dir / "Rectified_calibration" / f"{stem}.json"

            left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
            right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
            left_g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            pred = matcher.compute(left_g, right_g).astype(np.float32) / 16.0

            gt = cv2.imread(str(gt_disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            gt_depth = cv2.imread(str(gt_depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            calib = json.loads(calib_path.read_text())
            p1 = np.array(calib["P1"]["data"], dtype=np.float32).reshape(3, 4)
            p2 = np.array(calib["P2"]["data"], dtype=np.float32).reshape(3, 4)
            f = p1[0, 0]
            baseline_mm = abs(p2[0, 3] / p2[0, 0])
            pred_depth = f * baseline_mm / np.maximum(pred, 1e-6)
            mask = (gt > 0) & (gt_depth > 0) & (pred > 0) & np.isfinite(pred_depth)
            row = {"experiment": exp, "frame": stem}
            row.update(metrics(pred, gt, mask))
            row.update(depth_metrics(pred_depth, gt_depth, mask))
            rows.append(row)

            max_vis = np.percentile(gt[gt > 0], 99)
            pred_vis = vis_disparity(pred.clip(0), min_val=0, max_val=max_vis)
            gt_vis = vis_disparity(gt, min_val=0, max_val=max_vis)
            err = np.abs(pred - gt)
            err_vis = vis_disparity(err, min_val=0, max_val=min(20, np.percentile(err[mask], 99)))
            triptych = np.concatenate([left[..., ::-1], pred_vis, gt_vis, err_vis], axis=1)
            cv2.imwrite(str(out_dir / f"{exp}_{stem}_left_pred_gt_err.png"), triptych[..., ::-1])
            montage_rows.append((f"{exp}/{stem}", triptych))

    keys = [
        "experiment", "frame", "valid_px",
        "mae_px", "rmse_px", "bad1_pct", "bad2_pct", "bad5_pct",
        "depth_mae_mm", "depth_rmse_mm", "depth_bad1mm_pct", "depth_bad2mm_pct", "depth_bad5mm_pct",
    ]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    numeric_keys = keys[2:]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in numeric_keys}
    summary["frames"] = len(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    thumbs = []
    for label, triptych in montage_rows:
        thumb = cv2.resize(triptych, (1440, 288), interpolation=cv2.INTER_AREA)
        canvas = np.full((312, 1440, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[24:] = thumb
        thumbs.append(canvas)
    cv2.imwrite(str(out_dir / "montage_left_pred_gt_err.png"), np.concatenate(thumbs, axis=0)[..., ::-1])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
