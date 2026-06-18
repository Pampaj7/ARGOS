#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch


MODEL_CONFIG = {
    "S": {"feature_channels": 128, "n_transformer": 1},
    "M": {"feature_channels": 192, "n_transformer": 2},
    "L": {"feature_channels": 256, "n_transformer": 3},
    "XL": {"feature_channels": 384, "n_transformer": 3},
}


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_calib(path: Path) -> tuple[float, float]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration: {path}")
    m1 = fs.getNode("M1").mat()
    t = fs.getNode("T").mat()
    fs.release()
    fx = float(m1[0, 0])
    baseline_mm = float(abs(t.reshape(-1)[0]))
    return fx, baseline_mm


def load_calib_full(path: Path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration: {path}")
    out = {name: fs.getNode(name).mat() for name in ["M1", "D1", "M2", "D2", "R", "T"]}
    fs.release()
    return out


def rectify_sample(left: np.ndarray, right: np.ndarray, xyz: np.ndarray, calib_path: Path):
    h, w = left.shape[:2]
    calib = load_calib_full(calib_path)
    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        calib["M1"],
        calib["D1"],
        calib["M2"],
        calib["D2"],
        (w, h),
        calib["R"],
        calib["T"].reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(calib["M1"], calib["D1"], r1, p1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(calib["M2"], calib["D2"], r2, p2, (w, h), cv2.CV_32FC1)
    left_r = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    right_r = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    z = xyz[..., 2].astype(np.float32)
    valid = (np.isfinite(xyz).all(axis=-1) & (z > 0)).astype(np.uint8)
    z_clean = np.where(valid > 0, z, 0).astype(np.float32)
    z_r = cv2.remap(z_clean, map1x, map1y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_r = cv2.remap(valid, map1x, map1y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    baseline_mm = float(abs(p2[0, 3] / p2[0, 0]))
    fx = float(p1[0, 0])
    disp_r = fx * baseline_mm / np.maximum(z_r, 1e-6)
    valid_r &= z_r > 0
    return left_r, right_r, disp_r.astype(np.float32), z_r.astype(np.float32), valid_r, fx, baseline_mm


def load_scared_gt(keyframe_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = tifffile.imread(keyframe_dir / "left_depth_map.tiff").astype(np.float32)
    z = xyz[..., 2]
    valid = np.isfinite(xyz).all(axis=-1) & (z > 0)
    fx, baseline_mm = load_calib(keyframe_dir / "endoscope_calibration.yaml")
    disp = fx * baseline_mm / np.maximum(z, 1e-6)
    return disp.astype(np.float32), z.astype(np.float32), valid


def colorize(x: np.ndarray, vmax: float | None = None, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros((*x.shape, 3), dtype=np.uint8)
    if vmax is None:
        lo = float(np.nanpercentile(x[finite], 1))
        hi = float(np.nanpercentile(x[finite], 99))
    else:
        lo, hi = 0.0, float(vmax)
    if hi <= lo:
        return np.zeros((*x.shape, 3), dtype=np.uint8)
    u8 = (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cmap)


def build_model(args, checkpoint: Path | None, device):
    sys.path.insert(0, str(args.s2m2_src))
    from s2m2.core.model.s2m2 import S2M2

    cfg = MODEL_CONFIG[args.model_type]
    model = S2M2(
        feature_channels=cfg["feature_channels"],
        dim_expansion=1,
        num_transformer=cfg["n_transformer"],
        use_positivity=True,
        refine_iter=args.refine_iter,
    )
    if checkpoint is None:
        ckpt_path = Path(args.weights_dir) / f"CH{cfg['feature_channels']}NTR{cfg['n_transformer']}.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.my_load_state_dict(ckpt["state_dict"])
    else:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.my_load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, left: np.ndarray, right: np.ndarray, args, device) -> np.ndarray:
    sys.path.insert(0, str(args.s2m2_src))
    from s2m2.core.utils.image_utils import image_crop, image_pad

    h0, w0 = left.shape[:2]
    scale = 1.0
    if args.max_width and w0 > args.max_width:
        scale = args.max_width / float(w0)
        new_size = (args.max_width, int(round(h0 * scale)))
        left = cv2.resize(left, new_size, interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, new_size, interpolation=cv2.INTER_LINEAR)

    left_t = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).float().to(device)
    right_t = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).float().to(device)
    h, w = left_t.shape[-2:]
    with torch.amp.autocast(enabled=device.type == "cuda", device_type=device.type, dtype=torch.float16):
        pred, _occ, _conf = model(image_pad(left_t, 32), image_pad(right_t, 32))
    pred = image_crop(pred, (h, w)).squeeze().float().cpu().numpy()
    if scale != 1.0:
        pred = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_LINEAR) / scale
    return np.clip(pred.astype(np.float32), 0, None)


def metrics(pred_disp, gt_disp, gt_depth, valid, fx, baseline_mm):
    pred_depth = fx * baseline_mm / np.maximum(pred_disp, 1e-6)
    mask = valid & np.isfinite(pred_disp) & np.isfinite(pred_depth) & (gt_disp > 0) & (gt_depth > 0)
    disp_err = np.abs(pred_disp[mask] - gt_disp[mask])
    depth_err = np.abs(pred_depth[mask] - gt_depth[mask])
    return {
        "valid_px": int(mask.sum()),
        "disp_mae_px": float(disp_err.mean()),
        "disp_rmse_px": float(np.sqrt((disp_err**2).mean())),
        "disp_bad1_pct": float((disp_err > 1.0).mean() * 100.0),
        "disp_bad2_pct": float((disp_err > 2.0).mean() * 100.0),
        "disp_bad5_pct": float((disp_err > 5.0).mean() * 100.0),
        "depth_mae_mm": float(depth_err.mean()),
        "depth_rmse_mm": float(np.sqrt((depth_err**2).mean())),
        "depth_bad1mm_pct": float((depth_err > 1.0).mean() * 100.0),
        "depth_bad2mm_pct": float((depth_err > 2.0).mean() * 100.0),
        "depth_bad5mm_pct": float((depth_err > 5.0).mean() * 100.0),
    }


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared_root", type=Path, default=Path("dataset/SCARED/curated/keyframes_gt_dataset8/dataset_8"))
    parser.add_argument("--s2m2_src", type=Path, default=Path("../../external/frame_stereo_repos/s2m2/src"))
    parser.add_argument("--weights_dir", type=Path, default=Path("../../external/frame_stereo_repos/s2m2/weights/pretrain_weights"))
    parser.add_argument("--finetuned_checkpoint", type=Path, required=True)
    parser.add_argument("--model_type", default="S", choices=["S", "M", "L", "XL"])
    parser.add_argument("--refine_iter", type=int, default=3)
    parser.add_argument("--max_width", type=int, default=640)
    parser.add_argument("--rectify", action="store_true")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_model = build_model(args, None, device)
    ft_model = build_model(args, args.finetuned_checkpoint, device)

    rows = []
    montage_rows = []
    for keyframe_dir in sorted(args.scared_root.glob("keyframe_*")):
        frame = keyframe_dir.name
        left = read_rgb(keyframe_dir / "Left_Image.png")
        right = read_rgb(keyframe_dir / "Right_Image.png")
        if args.rectify:
            xyz = tifffile.imread(keyframe_dir / "left_depth_map.tiff").astype(np.float32)
            left, right, gt_disp, gt_depth, valid, fx, baseline_mm = rectify_sample(
                left,
                right,
                xyz,
                keyframe_dir / "endoscope_calibration.yaml",
            )
        else:
            gt_disp, gt_depth, valid = load_scared_gt(keyframe_dir)
            fx, baseline_mm = load_calib(keyframe_dir / "endoscope_calibration.yaml")

        base_pred = predict(base_model, left, right, args, device)
        ft_pred = predict(ft_model, left, right, args, device)
        for name, pred in [("pretrained", base_pred), ("servct_finetuned", ft_pred)]:
            row = {"run": name, "frame": frame, "fx": fx, "baseline_mm": baseline_mm}
            row.update(metrics(pred, gt_disp, gt_depth, valid, fx, baseline_mm))
            rows.append(row)
            np.save(args.out_dir / f"{name}_{frame}_disp.npy", pred)

        mask = valid & (gt_disp > 0)
        vmax = float(np.nanpercentile(gt_disp[mask], 99))
        err_vmax = min(40.0, float(np.nanpercentile(np.abs(base_pred[mask] - gt_disp[mask]), 99)))
        tiles = [
            cv2.cvtColor(left, cv2.COLOR_RGB2BGR),
            colorize(base_pred, vmax),
            colorize(ft_pred, vmax),
            colorize(gt_disp, vmax),
            colorize(np.abs(base_pred - gt_disp), err_vmax, cv2.COLORMAP_MAGMA),
            colorize(np.abs(ft_pred - gt_disp), err_vmax, cv2.COLORMAP_MAGMA),
        ]
        labels = ["left", "pretrained pred", "servct-ft pred", "gt disp", "pretrained err", "servct-ft err"]
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (240, 192), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
        row_img = np.concatenate(small, axis=1)
        cv2.putText(row_img, frame, (6, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        montage_rows.append(row_img)

    write_csv(args.out_dir / "metrics.csv", rows)
    summary = {}
    numeric = [k for k in rows[0] if k not in {"run", "frame"}]
    for run in sorted(set(r["run"] for r in rows)):
        rr = [r for r in rows if r["run"] == run]
        summary[run] = {k: float(np.mean([r[k] for r in rr])) for k in numeric}
        summary[run]["frames"] = len(rr)
    summary["delta_servct_finetuned_minus_pretrained"] = {
        k: summary["servct_finetuned"][k] - summary["pretrained"][k]
        for k in summary["pretrained"]
        if k != "frames"
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cv2.imwrite(str(args.out_dir / "scared_s2m2_base_vs_servct_finetuned_montage.png"), np.concatenate(montage_rows, axis=0))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
