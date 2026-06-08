import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from eval_metrics import failure_aware_metrics, summarize_rows


def load_gt(ref_dir, stem):
    disp_npy = ref_dir / "Disparity_float32" / f"{stem}.npy"
    depth_npy = ref_dir / "DepthL_float32" / f"{stem}.npy"
    mask_npy = ref_dir / "ValidMask" / f"{stem}.npy"
    if disp_npy.exists() and depth_npy.exists():
        gt = np.load(disp_npy).astype(np.float32)
        gt_depth = np.load(depth_npy).astype(np.float32)
        if mask_npy.exists():
            gt_mask = np.load(mask_npy).astype(bool)
        else:
            gt_mask = (gt > 0) & (gt_depth > 0) & np.isfinite(gt) & np.isfinite(gt_depth)
        return gt, gt_depth, gt_mask, "float32_npy"

    gt = cv2.imread(str(ref_dir / "Disparity" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
    gt_depth = cv2.imread(str(ref_dir / "DepthL" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
    gt_mask = (gt > 0) & (gt_depth > 0) & np.isfinite(gt) & np.isfinite(gt_depth)
    return gt, gt_depth, gt_mask, "uint16_png"


def make_sgbm(max_disp, block_size):
    max_disp = int(np.ceil(max_disp / 16.0) * 16)
    p1 = 8 * block_size * block_size
    p2 = 32 * block_size * block_size
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


def disp_vis(values, max_val):
    x = np.clip(values.astype(np.float32), 0, max(max_val, 1e-6))
    x = (x / max(max_val, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared_root", default="stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes")
    parser.add_argument("--out_dir", default="results/scared_sgbm_eval")
    parser.add_argument("--block_size", type=int, default=3)
    parser.add_argument("--max_disp", type=int, default=320)
    args = parser.parse_args()

    root = Path(args.scared_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matcher = make_sgbm(args.max_disp, args.block_size)

    rows = []
    montage_rows = []
    for exp_dir in sorted(root.glob("dataset_*")):
        ref_dir = exp_dir / "Reference_SCARED"
        for left_path in sorted((exp_dir / "Left_rectified").glob("*.png")):
            stem = left_path.stem
            right_path = exp_dir / "Right_rectified" / f"{stem}.png"
            calib_path = exp_dir / "Rectified_calibration" / f"{stem}.json"
            if not all(p.exists() for p in [right_path, calib_path]):
                continue

            left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
            right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
            left_g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            t0 = time.perf_counter()
            pred = matcher.compute(left_g, right_g).astype(np.float32) / 16.0
            runtime_ms = (time.perf_counter() - t0) * 1000.0

            gt, gt_depth, gt_mask, gt_source = load_gt(ref_dir, stem)
            calib = json.loads(calib_path.read_text())
            p1 = np.array(calib["P1"]["data"], dtype=np.float32).reshape(3, 4)
            p2 = np.array(calib["P2"]["data"], dtype=np.float32).reshape(3, 4)
            f = p1[0, 0]
            baseline_mm = abs(p2[0, 3] / p2[0, 0])
            pred_depth = f * baseline_mm / np.maximum(pred, 1e-6)
            mask = gt_mask & (pred > 0) & np.isfinite(pred_depth)
            if mask.sum() == 0:
                continue

            row = {"dataset": exp_dir.name, "frame": stem, "gt_source": gt_source, "runtime_ms": runtime_ms}
            row.update(failure_aware_metrics(pred, pred_depth, gt, gt_depth, gt_mask, mask))
            rows.append(row)

            if len(montage_rows) < 18:
                max_disp = np.percentile(gt[gt > 0], 99)
                err = np.abs(pred - gt)
                err_max = min(50, np.percentile(err[mask], 99))
                triptych = np.concatenate(
                    [left, disp_vis(pred, max_disp), disp_vis(gt, max_disp), disp_vis(err, err_max)],
                    axis=1,
                )
                montage_rows.append((f"{exp_dir.name}/{stem}", triptych))

    if not rows:
        raise RuntimeError("No SCARED converted frames found")

    keys = list(rows[0].keys())
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if montage_rows:
        thumbs = []
        for label, triptych in montage_rows:
            thumb = cv2.resize(triptych, (1440, 288), interpolation=cv2.INTER_AREA)
            canvas = np.full((312, 1440, 3), 255, dtype=np.uint8)
            cv2.putText(canvas, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            canvas[24:] = thumb
            thumbs.append(canvas)
        cv2.imwrite(str(out_dir / "montage_left_pred_gt_err.png"), np.concatenate(thumbs, axis=0))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
