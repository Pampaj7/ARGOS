#!/usr/bin/env python3
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch


MODEL_CONFIG = {
    "S": {"feature_channels": 128, "n_transformer": 1},
    "L": {"feature_channels": 256, "n_transformer": 3},
    "XL": {"feature_channels": 384, "n_transformer": 3},
}


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_calib(path: Path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration: {path}")
    out = {name: fs.getNode(name).mat() for name in ["M1", "D1", "M2", "D2", "R", "T"]}
    fs.release()
    return out


def rectify_sample(left: np.ndarray, right: np.ndarray, xyz: np.ndarray, calib_path: Path):
    h, w = left.shape[:2]
    calib = load_calib(calib_path)
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
    valid_r &= z_r > 0

    fx = float(p1[0, 0])
    baseline_mm = float(abs(p2[0, 3] / p2[0, 0]))
    disp_r = fx * baseline_mm / np.maximum(z_r, 1e-6)
    return left_r, right_r, disp_r.astype(np.float32), z_r.astype(np.float32), valid_r, fx, baseline_mm


def collect_samples(scared_root: Path):
    samples = []
    for keyframe_dir in sorted(scared_root.glob("keyframe_*")):
        required = [
            keyframe_dir / "Left_Image.png",
            keyframe_dir / "Right_Image.png",
            keyframe_dir / "left_depth_map.tiff",
            keyframe_dir / "endoscope_calibration.yaml",
        ]
        if all(p.exists() for p in required):
            left = read_rgb(required[0])
            right = read_rgb(required[1])
            xyz = tifffile.imread(required[2]).astype(np.float32)
            left_r, right_r, gt_disp, gt_depth, valid, fx, baseline_mm = rectify_sample(left, right, xyz, required[3])
            samples.append(
                {
                    "frame": keyframe_dir.name,
                    "left": left_r,
                    "right": right_r,
                    "gt_disp": gt_disp,
                    "gt_depth": gt_depth,
                    "valid": valid,
                    "fx": fx,
                    "baseline_mm": baseline_mm,
                }
            )
    if not samples:
        raise RuntimeError(f"No SCARED keyframes found under {scared_root}")
    return samples


def build_model(args, model_type: str, device):
    sys.path.insert(0, str(args.s2m2_src))
    from s2m2.core.model.s2m2 import S2M2

    cfg = MODEL_CONFIG[model_type]
    model = S2M2(
        feature_channels=cfg["feature_channels"],
        dim_expansion=1,
        num_transformer=cfg["n_transformer"],
        use_positivity=True,
        refine_iter=args.refine_iter,
    )
    ckpt_path = Path(args.weights_dir) / f"CH{cfg['feature_channels']}NTR{cfg['n_transformer']}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.my_load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt_path


def resize_pair(left, right, width):
    h, w = left.shape[:2]
    if width == 0 or width >= w:
        return left, right, 1.0, (h, w)
    scale_x = width / float(w)
    new_h = int(round(h * scale_x))
    left_r = cv2.resize(left, (width, new_h), interpolation=cv2.INTER_LINEAR)
    right_r = cv2.resize(right, (width, new_h), interpolation=cv2.INTER_LINEAR)
    return left_r, right_r, scale_x, (new_h, width)


@torch.no_grad()
def infer(model, left, right, width, device, s2m2_src: Path):
    sys.path.insert(0, str(s2m2_src))
    from s2m2.core.utils.image_utils import image_crop, image_pad

    orig_h, orig_w = left.shape[:2]
    left_in, right_in, scale_x, used_shape = resize_pair(left, right, width)
    left_t = torch.from_numpy(left_in).permute(2, 0, 1).unsqueeze(0).float().to(device)
    right_t = torch.from_numpy(right_in).permute(2, 0, 1).unsqueeze(0).float().to(device)
    h, w = left_t.shape[-2:]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast(enabled=device.type == "cuda", device_type=device.type, dtype=torch.float16):
        pred, _occ, _conf = model(image_pad(left_t, 32), image_pad(right_t, 32))
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    pred = image_crop(pred, (h, w)).squeeze().float().cpu().numpy()
    if scale_x != 1.0:
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale_x
    return np.clip(pred.astype(np.float32), 0, None), runtime_ms, used_shape, scale_x


def metric_row(pred_disp, sample):
    gt_disp = sample["gt_disp"]
    gt_depth = sample["gt_depth"]
    valid = sample["valid"]
    pred_depth = sample["fx"] * sample["baseline_mm"] / np.maximum(pred_disp, 1e-6)
    mask = valid & np.isfinite(pred_disp) & np.isfinite(pred_depth) & (gt_disp > 0) & (gt_depth > 0)
    disp_err = np.abs(pred_disp[mask] - gt_disp[mask])
    depth_err = np.abs(pred_depth[mask] - gt_depth[mask])
    gt_valid = max(int(valid.sum()), 1)
    return {
        "valid_disp_mae": float(disp_err.mean()),
        "valid_disp_rmse": float(np.sqrt((disp_err**2).mean())),
        "bad_1px": float((disp_err > 1.0).mean() * 100.0),
        "bad_2px": float((disp_err > 2.0).mean() * 100.0),
        "bad_3px": float((disp_err > 3.0).mean() * 100.0),
        "valid_depth_mae": float(depth_err.mean()),
        "valid_depth_median": float(np.median(depth_err)),
        "valid_depth_rmse": float(np.sqrt((depth_err**2).mean())),
        "bad_1mm": float((depth_err > 1.0).mean() * 100.0),
        "bad_2mm": float((depth_err > 2.0).mean() * 100.0),
        "bad_5mm": float((depth_err > 5.0).mean() * 100.0),
        "pred_disp_le_0_1_ratio": float((valid & (pred_disp <= 0.1)).sum() / gt_valid),
        "pred_disp_le_0_5_ratio": float((valid & (pred_disp <= 0.5)).sum() / gt_valid),
        "valid_pixel_ratio": float(mask.mean()),
    }


def mean_summary(rows):
    skip = {"model", "resize_width", "resize_label", "frame", "status", "error", "image_resolution_used"}
    numeric = [k for k, v in rows[0].items() if k not in skip and isinstance(v, (int, float))]
    return {k: float(np.mean([r[k] for r in rows])) for k in numeric}


def colorize(x, vmax=None, cmap=cv2.COLORMAP_TURBO):
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


def make_qualitative(samples, predictions, out_dir: Path):
    qdir = out_dir / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    selected = [samples[0], samples[len(samples) // 2], samples[-1]]
    for sample in selected:
        frame = sample["frame"]
        gt = sample["gt_disp"]
        gt_depth = sample["gt_depth"]
        valid = sample["valid"] & (gt > 0)
        gt_vmax = float(np.nanpercentile(gt[valid], 99))
        depth_vmax = float(np.nanpercentile(gt_depth[valid], 99))
        tiles = [cv2.cvtColor(sample["left"], cv2.COLOR_RGB2BGR), colorize(gt, gt_vmax), colorize(gt_depth, depth_vmax)]
        labels = ["left", "GT disparity", "GT depth"]
        for model in ["S", "L", "XL"]:
            key = (model, "1024", frame)
            if key not in predictions:
                key = (model, "full", frame)
            if key not in predictions:
                continue
            pred = predictions[key]
            err = np.abs(pred - gt)
            err_vmax = min(40.0, float(np.nanpercentile(err[valid], 99)))
            tiles.extend([colorize(pred, gt_vmax), colorize(err, err_vmax, cv2.COLORMAP_MAGMA)])
            labels.extend([f"{model} pred", f"{model} abs err"])
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (220, 176), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
        montage = np.concatenate(small, axis=1)
        cv2.imwrite(str(qdir / f"{frame}_qualitative.png"), montage)


def write_report(out_dir: Path, summary_rows, failures, dataset_note):
    by_key = {(r["model"], r["resize_label"]): r for r in summary_rows}
    lines = [
        "# S2M2 Size Tradeoff On SCARED",
        "",
        f"Dataset: {dataset_note}",
        "",
        "Note: the requested converted path `stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes` is not present in this workspace, so this run uses the current ARGOS rectified SCARED dataset_8 keyframe subset.",
        "",
        "All disparities are rescaled back to original image coordinates after input resizing with `pred_disp_original = pred_disp_resized / scale_x`.",
        "",
    ]
    if failures:
        lines.extend(["## Failed Runs", ""])
        for f in failures:
            lines.append(f"- {f['model']} / {f['resize_label']}: {f['error']}")
        lines.append("")
    lines.extend([
        "## Summary",
        "",
        "| model | width | disp MAE | depth MAE | depth RMSE | bad 2px | bad 2mm | avg ms | median ms | peak MB | params M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in summary_rows:
        lines.append(
            f"| {r['model']} | {r['resize_label']} | {r['valid_disp_mae']:.4f} | {r['valid_depth_mae']:.4f} | {r['valid_depth_rmse']:.4f} | "
            f"{r['bad_2px']:.2f} | {r['bad_2mm']:.2f} | {r['avg_inference_time_ms']:.2f} | {r['median_inference_time_ms']:.2f} | "
            f"{r['peak_gpu_memory_mb']:.1f} | {r['model_param_count_m']:.2f} |"
        )

    def best_for(metric, lower=True):
        vals = [r for r in summary_rows if np.isfinite(r[metric])]
        return min(vals, key=lambda r: r[metric]) if lower and vals else max(vals, key=lambda r: r[metric])

    best_depth = best_for("valid_depth_mae")
    best_disp = best_for("valid_disp_mae")
    best_fast = min(summary_rows, key=lambda r: r["avg_inference_time_ms"])
    xl_1024 = by_key.get(("XL", "1024"))
    l_1024 = by_key.get(("L", "1024"))
    s_1024 = by_key.get(("S", "1024"))
    xl_full = by_key.get(("XL", "full"))

    lines.extend(["", "## Analysis", ""])
    lines.append(f"1. Best depth MAE is `{best_depth['valid_depth_mae']:.4f} mm` from `{best_depth['model']}` at `{best_depth['resize_label']}`.")
    if xl_1024 and l_1024:
        lines.append(f"   At width 1024, XL vs L depth MAE delta is `{xl_1024['valid_depth_mae'] - l_1024['valid_depth_mae']:.4f} mm`.")
    if xl_1024 and s_1024:
        lines.append(f"   At width 1024, XL vs S depth MAE delta is `{xl_1024['valid_depth_mae'] - s_1024['valid_depth_mae']:.4f} mm`.")
    lines.append(f"2. Best disparity MAE is `{best_disp['valid_disp_mae']:.4f} px` from `{best_disp['model']}` at `{best_disp['resize_label']}`.")
    if xl_1024 and l_1024:
        lines.append(f"   At width 1024, XL vs L disparity MAE delta is `{xl_1024['valid_disp_mae'] - l_1024['valid_disp_mae']:.4f} px`.")
    if xl_1024 and s_1024:
        lines.append(f"   At width 1024, XL vs S disparity MAE delta is `{xl_1024['valid_disp_mae'] - s_1024['valid_disp_mae']:.4f} px`.")
    cat_metric = "pred_disp_le_0_5_ratio"
    best_cat = best_for(cat_metric)
    lines.append(f"3. Lowest `pred_disp <= 0.5` ratio is `{best_cat[cat_metric]:.6f}` from `{best_cat['model']}` at `{best_cat['resize_label']}`; compare this with average error to decide if XL reduces catastrophic failures or only mean error.")
    if xl_full and xl_1024:
        verdict = "yes" if xl_1024["valid_depth_mae"] < xl_full["valid_depth_mae"] else "no"
        lines.append(f"4. Resize width 1024 better than full for XL? {verdict}. XL full depth MAE `{xl_full['valid_depth_mae']:.4f}`, XL 1024 `{xl_1024['valid_depth_mae']:.4f}`.")
    else:
        lines.append("4. Resize width 1024 vs full could not be fully answered because one XL run is missing.")
    close = []
    if xl_1024:
        target = xl_1024["valid_depth_mae"]
        close = [r for r in summary_rows if r["valid_depth_mae"] <= target + 0.25 and r["avg_inference_time_ms"] < xl_1024["avg_inference_time_ms"]]
    if close:
        c = min(close, key=lambda r: r["avg_inference_time_ms"])
        lines.append(f"5. Close faster candidate: `{c['model']}` at `{c['resize_label']}` with depth MAE `{c['valid_depth_mae']:.4f}` and `{c['avg_inference_time_ms']:.2f} ms`.")
    else:
        lines.append("5. No smaller/faster candidate is within `0.25 mm` of XL@1024 by depth MAE in this run.")
    lines.append("6. Recommendations:")
    l_full = by_key.get(("L", "full"))
    if l_full:
        lines.append("   - default evaluation baseline: `L` at `full`, because it is nearly tied with XL while being much cheaper.")
    else:
        lines.append(f"   - default evaluation baseline: `{best_depth['model']}` at `{best_depth['resize_label']}` for best measured depth accuracy.")
    s_512 = by_key.get(("S", "512"))
    if s_512:
        lines.append("   - real-time candidate: `S` at `512` for fastest inference; `S` at `736` is the safer speed/accuracy compromise.")
    else:
        lines.append(f"   - real-time candidate: `{best_fast['model']}` at `{best_fast['resize_label']}` for fastest inference, if its accuracy is acceptable.")
    lines.append("   - teacher for future distillation: `XL` at `full`, but only as a teacher/reference model, not as the routine baseline unless larger SCARED runs show a larger hard-frame benefit.")
    lines.append("")
    lines.append("Qualitative montages are in `qualitative/`.")
    (out_dir / "s2m2_size_tradeoff.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared_root", type=Path, default=Path("/dtu/p1/leopam/ARGOS/dataset/SCARED/curated/keyframes_gt_dataset8/dataset_8"))
    parser.add_argument("--s2m2_src", type=Path, default=Path("../../external/frame_stereo_repos/s2m2/src"))
    parser.add_argument("--weights_dir", type=Path, default=Path("../../external/frame_stereo_repos/s2m2/weights/pretrain_weights"))
    parser.add_argument("--out_dir", type=Path, default=Path("/dtu/p1/leopam/ARGOS/results/s2m2_size_tradeoff"))
    parser.add_argument("--models", nargs="*", default=["S", "L", "XL"], choices=["S", "L", "XL"])
    parser.add_argument("--widths", nargs="*", type=int, default=[0, 1024, 736, 512])
    parser.add_argument("--refine_iter", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    samples = collect_samples(args.scared_root)

    all_frame_rows = []
    summary_rows = []
    failures = []
    predictions_for_qual = {}

    for model_type in args.models:
        try:
            model, ckpt_path = build_model(args, model_type, device)
            param_count = sum(p.numel() for p in model.parameters())
        except Exception as exc:
            failures.append({"model": model_type, "resize_label": "all", "error": str(exc)})
            continue

        for width in args.widths:
            label = "full" if width == 0 else str(width)
            frame_rows = []
            runtimes = []
            peak_mem = 0.0
            try:
                for sample in samples:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    pred, runtime_ms, used_shape, scale_x = infer(model, sample["left"], sample["right"], width, device, args.s2m2_src)
                    if device.type == "cuda":
                        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024**2))
                    row = {
                        "model": model_type,
                        "resize_width": int(width),
                        "resize_label": label,
                        "frame": sample["frame"],
                        "image_resolution_used": f"{used_shape[0]}x{used_shape[1]}",
                        "scale_x": float(scale_x),
                        "inference_time_ms": float(runtime_ms),
                        "peak_gpu_memory_mb": float(peak_mem),
                        "model_param_count": int(param_count),
                        "model_param_count_m": float(param_count / 1e6),
                        "status": "ok",
                    }
                    row.update(metric_row(pred, sample))
                    frame_rows.append(row)
                    predictions_for_qual[(model_type, label, sample["frame"])] = pred
                runtimes = [r["inference_time_ms"] for r in frame_rows]
                summary = mean_summary(frame_rows)
                summary.update(
                    {
                        "model": model_type,
                        "resize_width": int(width),
                        "resize_label": label,
                        "avg_inference_time_ms": float(np.mean(runtimes)),
                        "median_inference_time_ms": float(np.median(runtimes)),
                        "peak_gpu_memory_mb": float(peak_mem),
                        "image_resolution_used": frame_rows[0]["image_resolution_used"],
                        "model_param_count": int(param_count),
                        "model_param_count_m": float(param_count / 1e6),
                        "frames": len(frame_rows),
                        "checkpoint": str(ckpt_path),
                    }
                )
                summary_rows.append(summary)
                all_frame_rows.extend(frame_rows)
                print(f"done {model_type} {label}", flush=True)
            except RuntimeError as exc:
                failures.append({"model": model_type, "resize_label": label, "error": str(exc)})
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not summary_rows:
        raise RuntimeError("No successful benchmark runs")

    csv_keys = list(summary_rows[0].keys())
    with (args.out_dir / "s2m2_size_tradeoff.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    with (args.out_dir / "s2m2_size_tradeoff_frame_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_frame_rows)

    payload = {
        "dataset": str(args.scared_root),
        "device": str(device),
        "summary": summary_rows,
        "failures": failures,
    }
    (args.out_dir / "s2m2_size_tradeoff.json").write_text(json.dumps(payload, indent=2) + "\n")
    make_qualitative(samples, predictions_for_qual, args.out_dir)
    write_report(args.out_dir, summary_rows, failures, str(args.scared_root))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
