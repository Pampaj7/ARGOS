import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from s2m2.core.model.s2m2 import S2M2
from s2m2.core.utils.image_utils import image_crop, image_pad


MODEL_CONFIG = {
    "S": {"feature_channels": 128, "n_transformer": 1},
    "M": {"feature_channels": 192, "n_transformer": 2},
    "L": {"feature_channels": 256, "n_transformer": 3},
    "XL": {"feature_channels": 384, "n_transformer": 3},
}


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


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


def disp_vis(x, max_val=None):
    x = x.astype(np.float32)
    if max_val is not None:
        x = np.clip(x, 0, max_val)
    denom = max(float(x.max()), 1e-6)
    x = (x / denom * 255.0).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)


def load_custom_checkpoint(checkpoint_path, model_type, use_positivity, refine_iter, device):
    cfg = MODEL_CONFIG[model_type]
    model = S2M2(
        feature_channels=cfg["feature_channels"],
        dim_expansion=1,
        num_transformer=cfg["n_transformer"],
        use_positivity=use_positivity,
        refine_iter=refine_iter,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.my_load_state_dict(state_dict)
    return model.to(device).eval()


def load_pretrained_no_open3d(pretrain_path, model_type, use_positivity, refine_iter, device):
    cfg = MODEL_CONFIG[model_type]
    model = S2M2(
        feature_channels=cfg["feature_channels"],
        dim_expansion=1,
        num_transformer=cfg["n_transformer"],
        use_positivity=use_positivity,
        refine_iter=refine_iter,
    )
    ckpt_path = Path(pretrain_path) / f"CH{cfg['feature_channels']}NTR{cfg['n_transformer']}.pth"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.my_load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def run_stereo_matching_no_open3d(model, left_torch, right_torch, device):
    h, w = left_torch.shape[-2:]
    left_pad = image_pad(left_torch, 32).to(device)
    right_pad = image_pad(right_torch, 32).to(device)
    with torch.amp.autocast(enabled=device.type == "cuda", device_type=device.type, dtype=torch.float16):
        pred_disp, pred_occ, pred_conf = model(left_pad, right_pad)
    pred_disp = image_crop(pred_disp, (h, w)).squeeze().float()
    pred_occ = image_crop(pred_occ, (h, w)).squeeze().float()
    pred_conf = image_crop(pred_conf, (h, w)).squeeze().float()
    margin = 100
    avg_conf = pred_conf[margin:-margin, margin:-margin].mean().item()
    return pred_disp, pred_occ, pred_conf, avg_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--servct_root", default="../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT")
    parser.add_argument("--weights_dir", default="weights/pretrain_weights")
    parser.add_argument("--checkpoint", default=None, help="Optional fine-tuned checkpoint to load instead of weights_dir")
    parser.add_argument("--model_type", default="S", choices=["S", "M", "L", "XL"])
    parser.add_argument("--num_refine", type=int, default=3)
    parser.add_argument("--reference", choices=["Reference_CT", "Reference_RGB"], default="Reference_CT")
    parser.add_argument("--out_dir", default="output_servct_eval_s2m2_S")
    parser.add_argument("--allow_negative", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        model = load_custom_checkpoint(args.checkpoint, args.model_type, not args.allow_negative, args.num_refine, device)
    else:
        model = load_pretrained_no_open3d(
            args.weights_dir,
            args.model_type,
            use_positivity=not args.allow_negative,
            refine_iter=args.num_refine,
            device=device,
        )
    if model is None:
        raise RuntimeError("Failed to load S2M2 model")

    root = Path(args.servct_root)
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
            if not (right_path.exists() and gt_disp_path.exists() and gt_depth_path.exists() and calib_path.exists()):
                continue

            left = load_rgb(left_path)
            right = load_rgb(right_path)
            left_t = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).to(device)
            right_t = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).to(device)

            pred_disp, pred_occ, pred_conf, avg_conf = run_stereo_matching_no_open3d(model, left_t, right_t, device)
            pred = pred_disp.detach().cpu().numpy().astype(np.float32)
            conf = pred_conf.detach().cpu().numpy().astype(np.float32)
            occ = pred_occ.detach().cpu().numpy().astype(np.float32)

            gt = cv2.imread(str(gt_disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            gt_depth = cv2.imread(str(gt_depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            calib = json.loads(calib_path.read_text())
            p1 = np.array(calib["P1"]["data"], dtype=np.float32).reshape(3, 4)
            p2 = np.array(calib["P2"]["data"], dtype=np.float32).reshape(3, 4)
            f = p1[0, 0]
            baseline_mm = abs(p2[0, 3] / p2[0, 0])
            pred_depth = f * baseline_mm / np.maximum(pred, 1e-6)

            mask = (gt > 0) & (gt_depth > 0) & np.isfinite(pred_depth)
            row = {
                "experiment": exp,
                "frame": stem,
                "avg_conf": float(avg_conf),
                "valid_conf_occ_pct": float(((conf > 0.1) & (occ > 0.5)).mean() * 100.0),
            }
            row.update(disp_metrics(pred, gt, mask))
            row.update(depth_metrics(pred_depth, gt_depth, mask))
            rows.append(row)

            max_disp = np.percentile(gt[mask], 99)
            pred_v = disp_vis(pred, max_disp)
            gt_v = disp_vis(gt, max_disp)
            err_v = disp_vis(np.abs(pred - gt), min(20, np.percentile(np.abs(pred[mask] - gt[mask]), 99)))
            triptych = np.concatenate([left, pred_v[..., ::-1], gt_v[..., ::-1], err_v[..., ::-1]], axis=1)
            cv2.imwrite(str(out_dir / f"{exp}_{stem}_left_pred_gt_err.png"), triptych[..., ::-1])
            montage_rows.append((f"{exp}/{stem}", triptych))

    if not rows:
        raise RuntimeError("No SERV-CT frames found")

    keys = [
        "experiment", "frame", "avg_conf", "valid_conf_occ_pct", "valid_px",
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
    montage = np.concatenate(thumbs, axis=0)
    cv2.imwrite(str(out_dir / "montage_left_pred_gt_err.png"), montage[..., ::-1])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
