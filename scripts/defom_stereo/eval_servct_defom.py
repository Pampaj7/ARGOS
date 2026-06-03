import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append("core")
from core.defom_stereo import DEFOMStereo
from utils.utils import InputPadder


def load_image(path, device):
    img = np.array(Image.open(path)).astype(np.uint8)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return img, tensor[None].to(device)


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


def disp_vis(x, max_val):
    x = np.clip(x.astype(np.float32), 0, max(max_val, 1e-6))
    x = (x / max(max_val, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)[..., ::-1]


def load_model(args, device):
    model = DEFOMStereo(args)
    checkpoint = torch.load(args.restore_ckpt, map_location="cpu")
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--servct_root", required=True)
    parser.add_argument("--restore_ckpt", required=True)
    parser.add_argument("--out_dir", default="output_servct_eval_defom")
    parser.add_argument("--reference", choices=["Reference_CT", "Reference_RGB"], default="Reference_CT")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--valid_iters", type=int, default=16)
    parser.add_argument("--scale_iters", type=int, default=4)
    parser.add_argument("--dinov2_encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--idepth_scale", type=float, default=0.5)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128] * 3)
    parser.add_argument("--corr_implementation", choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--scale_list", type=float, nargs="+", default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--scale_corr_radius", type=int, default=2)
    parser.add_argument("--n_downsample", type=int, default=2, choices=[2, 3])
    parser.add_argument("--context_norm", type=str, default="batch", choices=["group", "batch", "instance", "none"])
    parser.add_argument("--n_gru_layers", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args, device)

    rows = []
    montage_rows = []
    root = Path(args.servct_root)
    with torch.no_grad():
        for exp in ["Experiment_1", "Experiment_2"]:
            exp_dir = root / exp
            ref_dir = exp_dir / args.reference
            for left_path in sorted((exp_dir / "Left_rectified").glob("*.png")):
                stem = left_path.stem
                right_path = exp_dir / "Right_rectified" / f"{stem}.png"
                gt_disp_path = ref_dir / "Disparity" / f"{stem}.png"
                gt_depth_path = ref_dir / "DepthL" / f"{stem}.png"
                calib_path = exp_dir / "Rectified_calibration" / f"{stem}.json"
                if not (right_path.exists() and gt_disp_path.exists() and gt_depth_path.exists() and calib_path.exists()):
                    continue

                left_np, image1 = load_image(left_path, device)
                _, image2 = load_image(right_path, device)
                padder = InputPadder(image1.shape, divis_by=32)
                image1, image2 = padder.pad(image1, image2)
                pred = model(image1, image2, iters=args.valid_iters, scale_iters=args.scale_iters, test_mode=True)
                pred = padder.unpad(pred).squeeze().float().cpu().numpy()

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
                err = np.abs(pred - gt)
                triptych = np.concatenate(
                    [left_np, disp_vis(pred, max_disp), disp_vis(gt, max_disp), disp_vis(err, min(20, np.percentile(err[mask], 99)))],
                    axis=1,
                )
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
        cv2.imwrite(str(out_dir / "montage_left_pred_gt_err.png"), np.concatenate(thumbs, axis=0)[..., ::-1])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
