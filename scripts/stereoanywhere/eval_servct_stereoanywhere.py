import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_DIR))

from models.stereoanywhere import StereoAnywhere
from models.depth_anything_v2 import get_depth_anything_v2


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(img, device):
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


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
    if max_val is None:
        max_val = np.percentile(x[np.isfinite(x)], 99) if np.isfinite(x).any() else 1.0
    x = np.clip(x, 0, max(max_val, 1e-6))
    x = (x / max(max_val, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)[..., ::-1]


def build_args(cli_args):
    return SimpleNamespace(
        maxdisp=cli_args.maxdisp,
        n_downsample=2,
        n_additional_hourglass=0,
        volume_channels=8,
        vol_downsample=0,
        vol_n_masks=8,
        use_truncate_vol=False,
        mirror_conf_th=0.98,
        mirror_attenuation=0.9,
        use_aggregate_stereo_vol=False,
        use_aggregate_mono_vol=False,
        normal_gain=10,
        lrc_th=1.0,
        mixed_precision=cli_args.mixed_precision,
        corr_implementation="reg",
    )


def load_stereo_model(weights_path, model_args, device):
    model = torch.nn.DataParallel(StereoAnywhere(model_args)).to(device).eval()
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model


def load_mono_model(weights_path, encoder, device):
    if not weights_path:
        return None
    return get_depth_anything_v2(weights_path, encoder=encoder).eval().to(device)


@torch.no_grad()
def infer(model, mono_model, left, right, model_args, cli_args, device):
    h, w = left.shape[:2]
    if cli_args.iscale != 1.0:
        new_w = round(w / cli_args.iscale)
        new_h = round(h / cli_args.iscale)
        left = cv2.resize(left, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    left_t = to_tensor(left, device)
    right_t = to_tensor(right, device)
    if mono_model is None:
        left_mono = torch.zeros_like(left_t[:, :1])
        right_mono = torch.zeros_like(right_t[:, :1])
    else:
        mono_depths = mono_model.infer_image(torch.cat([left_t, right_t], 0), input_size_width=518, input_size_height=518)
        mono_depths = (mono_depths - mono_depths.amin(dim=(-2, -1), keepdim=True)) / (
            mono_depths.amax(dim=(-2, -1), keepdim=True) - mono_depths.amin(dim=(-2, -1), keepdim=True) + 1e-6
        )
        left_mono = mono_depths[0:1]
        right_mono = mono_depths[1:2]

    ht, wt = left_t.shape[-2:]
    pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
    pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
    pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    left_t = F.pad(left_t, pad, mode="replicate")
    right_t = F.pad(right_t, pad, mode="replicate")
    left_mono = F.pad(left_mono, pad, mode="replicate")
    right_mono = F.pad(right_mono, pad, mode="replicate")

    with torch.autocast(device_type=device.type, enabled=cli_args.mixed_precision):
        pred_disps, _ = model(left_t, right_t, left_mono, right_mono, test_mode=True, iters=cli_args.iters)

    pred = pred_disps.squeeze().float().detach().cpu().numpy()
    ph, pw = pred.shape[-2:]
    pred = pred[pad[2]:ph - pad[3], pad[0]:pw - pad[1]]
    if np.count_nonzero(pred > 0) < np.count_nonzero((-pred) > 0):
        pred = -pred
    if cli_args.iscale != 1.0:
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR) * cli_args.iscale
    return np.clip(pred, 0, None).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--servct_root", default="../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT")
    parser.add_argument("--weights", default="weights/stereoanywhere_sceneflow.pth")
    parser.add_argument("--loadmonomodel", default="weights/depth_anything_v2_vits.pth")
    parser.add_argument("--vit_encoder", choices=["vits", "vitb", "vitl"], default="vits")
    parser.add_argument("--reference", choices=["Reference_CT", "Reference_RGB"], default="Reference_CT")
    parser.add_argument("--out_dir", default="output_servct_eval_stereoanywhere")
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--maxdisp", type=int, default=192)
    parser.add_argument("--iscale", type=float, default=1.0)
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_args = build_args(args)
    model = load_stereo_model(args.weights, model_args, device)
    mono_model = load_mono_model(args.loadmonomodel, args.vit_encoder, device)

    rows = []
    montage_rows = []
    root = Path(args.servct_root)
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
            pred = infer(model, mono_model, left, right, model_args, args, device)

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
                [left, disp_vis(pred, max_disp), disp_vis(gt, max_disp), disp_vis(err, min(20, np.percentile(err[mask], 99)))],
                axis=1,
            )
            cv2.imwrite(str(out_dir / f"{exp}_{stem}_left_pred_gt_err.png"), triptych[..., ::-1])
            montage_rows.append((f"{exp}/{stem}", triptych))

    if not rows:
        raise RuntimeError("No SERV-CT frames found")

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
    montage = np.concatenate(thumbs, axis=0)
    cv2.imwrite(str(out_dir / "montage_left_pred_gt_err.png"), montage[..., ::-1])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
