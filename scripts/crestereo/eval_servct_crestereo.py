import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from stereo_matching_crestereo import CrestereoMatching


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def apply_colormap(x, vmax=None):
    x = x.astype(np.float32)
    if vmax is None:
        vmax = np.percentile(x[np.isfinite(x)], 99) if np.isfinite(x).any() else 1.0
    x = np.clip(x / max(vmax, 1e-6), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]


def disp_metrics(pred, gt, mask):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--servct_root", default="../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT")
    parser.add_argument("--out_dir", default="output_servct_eval_crestereo")
    parser.add_argument("--input_hw", type=float, default=1.0)
    parser.add_argument("--max_disp", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = {"max_disp": args.max_disp}
    if args.input_hw != 1.0:
        cfg["input_hw"] = args.input_hw
    matcher = CrestereoMatching(cfg).to(device).eval()

    root = Path(args.servct_root)
    rows = []
    montage_rows = []
    for exp in ["Experiment_1", "Experiment_2"]:
        exp_dir = root / exp
        ref_dir = exp_dir / "Reference_CT"
        for left_path in sorted((exp_dir / "Left_rectified").glob("*.png")):
            stem = left_path.stem
            right_path = exp_dir / "Right_rectified" / f"{stem}.png"
            gt_disp_path = ref_dir / "Disparity" / f"{stem}.png"
            gt_depth_path = ref_dir / "DepthL" / f"{stem}.png"
            calib_path = exp_dir / "Rectified_calibration" / f"{stem}.json"
            if not (right_path.exists() and gt_disp_path.exists() and gt_depth_path.exists() and calib_path.exists()):
                continue

            left = load_rgb(left_path)
            right = load_rgb(right_path)
            pred = matcher(left, right)["disparity"].astype(np.float32)

            gt = cv2.imread(str(gt_disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            gt_depth = cv2.imread(str(gt_depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            calib = json.loads(calib_path.read_text())
            p1 = np.array(calib["P1"]["data"], dtype=np.float32).reshape(3, 4)
            p2 = np.array(calib["P2"]["data"], dtype=np.float32).reshape(3, 4)
            f = p1[0, 0]
            baseline_mm = abs(p2[0, 3] / p2[0, 0])
            pred_depth = f * baseline_mm / np.maximum(pred, 1e-6)

            mask = (gt > 0) & (gt_depth > 0) & np.isfinite(pred_depth)
            row = {"experiment": exp, "frame": stem}
            row.update(disp_metrics(pred, gt, mask))
            row.update(depth_metrics(pred_depth, gt_depth, mask))
            rows.append(row)

            max_disp = np.percentile(gt[mask], 99)
            pred_v = apply_colormap(pred, max_disp)
            gt_v = apply_colormap(gt, max_disp)
            err = np.abs(pred - gt)
            err_v = apply_colormap(err, min(20, np.percentile(err[mask], 99)))
            triptych = np.concatenate([left, pred_v, gt_v, err_v], axis=1)
            cv2.imwrite(str(out_dir / f"{exp}_{stem}_left_pred_gt_err.png"), triptych[..., ::-1])
            montage_rows.append((f"{exp}/{stem}", triptych))

    keys = [
        "experiment", "frame", "valid_px", "mae_px", "rmse_px", "bad1_pct", "bad2_pct", "bad5_pct",
        "depth_mae_mm", "depth_rmse_mm", "depth_bad1mm_pct", "depth_bad2mm_pct", "depth_bad5mm_pct",
    ]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys[2:]}
    summary["frames"] = len(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    thumbs = []
    for label, triptych in montage_rows:
        thumb = cv2.resize(triptych, (1440, 288), interpolation=cv2.INTER_AREA)
        canvas = np.full((312, 1440, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[24:] = thumb
        thumbs.append(canvas)
    if thumbs:
        montage = np.concatenate(thumbs, axis=0)
        cv2.imwrite(str(out_dir / "montage_left_pred_gt_err.png"), montage[..., ::-1])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
