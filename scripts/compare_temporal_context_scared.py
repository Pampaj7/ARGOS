#!/usr/bin/env python3
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


MODEL_CONFIG = {
    "S": {"feature_channels": 128, "n_transformer": 1},
    "M": {"feature_channels": 192, "n_transformer": 2},
    "L": {"feature_channels": 256, "n_transformer": 3},
    "XL": {"feature_channels": 384, "n_transformer": 3},
}


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_float(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(x.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def robust_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.nanpercentile(x[finite], [1, 99])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def colorize_norm(x: np.ndarray) -> np.ndarray:
    u8 = (robust_norm(x) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def colorize_error(x: np.ndarray) -> np.ndarray:
    u8 = (np.clip(x.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)


def temporal_error_maps(stack: np.ndarray) -> np.ndarray:
    norm = np.stack([robust_norm(frame) for frame in stack], axis=0)
    errors = np.zeros_like(norm, dtype=np.float32)
    if len(norm) > 1:
        errors[1:] = np.abs(np.diff(norm, axis=0))
    return errors


def temporal_metrics(stack: np.ndarray) -> dict:
    norm = np.stack([robust_norm(frame) for frame in stack], axis=0)
    diffs = np.abs(np.diff(norm, axis=0))
    means = [float(d.mean()) for d in diffs]
    p95s = [float(np.percentile(d, 95)) for d in diffs]
    return {
        "adjacent_delta_mean": float(np.mean(means)) if means else 0.0,
        "adjacent_delta_p95_mean": float(np.mean(p95s)) if p95s else 0.0,
        "adjacent_delta_per_pair": means,
        "frame_mean_std": float(np.std([frame.mean() for frame in norm])),
        "frame_p95_std": float(np.std([np.percentile(frame, 95) for frame in norm])),
    }


def load_fastfoundation_sequence(output_pattern: str, frames: int, target_shape: tuple[int, int]) -> np.ndarray:
    maps = []
    for idx in range(frames):
        depth_path = Path(output_pattern.format(idx=idx)) / "depth_meter.npy"
        if not depth_path.exists():
            raise RuntimeError(f"Missing FastFoundation depth output: {depth_path}")
        depth = np.load(depth_path).astype(np.float32)
        inv_depth = 1.0 / np.maximum(depth, 1e-6)
        maps.append(resize_float(inv_depth, target_shape))
    return np.stack(maps, axis=0)


def load_fastfoundation_stack(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim != 3:
        raise RuntimeError(f"Expected FastFoundation depth stack [T,H,W], got {depth.shape} from {path}")
    inv_depth = 1.0 / np.maximum(depth, 1e-6)
    inv_depth[~np.isfinite(inv_depth)] = 0.0
    return np.stack([resize_float(frame, target_shape) for frame in inv_depth], axis=0)


def load_stereoanyvideo_sequence(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    disp = np.load(path).astype(np.float32)
    if disp.ndim == 4:
        disp = disp[:, 0]
    return np.stack([resize_float(frame, target_shape) for frame in disp], axis=0)


def load_s2m2_model(args, device):
    sys.path.insert(0, str(args.s2m2_src))
    from s2m2.core.model.s2m2 import S2M2

    cfg = MODEL_CONFIG[args.s2m2_model_type]
    model = S2M2(
        feature_channels=cfg["feature_channels"],
        dim_expansion=1,
        num_transformer=cfg["n_transformer"],
        use_positivity=True,
        refine_iter=args.s2m2_refine_iter,
    )
    ckpt_path = args.s2m2_weights / f"CH{cfg['feature_channels']}NTR{cfg['n_transformer']}.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.my_load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def run_s2m2_sequence(args, left_paths, right_paths, target_shape: tuple[int, int], out_dir: Path) -> np.ndarray:
    sys.path.insert(0, str(args.s2m2_src))
    from s2m2.core.utils.image_utils import image_crop, image_pad

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_s2m2_model(args, device)
    maps = []
    runtimes = []
    for idx, (left_path, right_path) in enumerate(zip(left_paths, right_paths)):
        left = read_rgb(left_path)
        right = read_rgb(right_path)
        original_h, original_w = left.shape[:2]
        scale = 1.0
        if args.s2m2_max_width and original_w > args.s2m2_max_width:
            scale = args.s2m2_max_width / float(original_w)
            new_size = (args.s2m2_max_width, int(round(original_h * scale)))
            left = cv2.resize(left, new_size, interpolation=cv2.INTER_LINEAR)
            right = cv2.resize(right, new_size, interpolation=cv2.INTER_LINEAR)
        left_t = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).float().to(device)
        right_t = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).float().to(device)
        h, w = left_t.shape[-2:]
        t0 = time.perf_counter()
        with torch.amp.autocast(enabled=device.type == "cuda", device_type=device.type, dtype=torch.float16):
            pred, _occ, _conf = model(image_pad(left_t, 32), image_pad(right_t, 32))
        pred = image_crop(pred, (h, w)).squeeze().float().cpu().numpy()
        runtimes.append((time.perf_counter() - t0) * 1000.0)
        if scale != 1.0:
            pred = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_LINEAR) / scale
        pred = resize_float(np.clip(pred, 0, None), target_shape)
        np.save(out_dir / f"s2m2_disp_{idx:03d}.npy", pred)
        cv2.imwrite(str(out_dir / f"s2m2_disp_{idx:03d}.png"), colorize_norm(pred))
        maps.append(pred)
    (out_dir / "s2m2_runtime.json").write_text(json.dumps({"runtime_ms": runtimes, "mean_runtime_ms": float(np.mean(runtimes))}, indent=2) + "\n")
    return np.stack(maps, axis=0)


def make_montage(left_paths, model_maps: dict[str, np.ndarray], out_path: Path) -> None:
    rows = []
    frames = len(left_paths)
    model_names = list(model_maps.keys())
    for idx in range(frames):
        left = cv2.cvtColor(read_rgb(left_paths[idx]), cv2.COLOR_RGB2BGR)
        left = cv2.resize(left, (320, 192), interpolation=cv2.INTER_AREA)
        tiles = [left]
        for name in model_names:
            color = colorize_norm(model_maps[name][idx])
            color = cv2.resize(color, (320, 192), interpolation=cv2.INTER_AREA)
            cv2.putText(color, name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(color)
        row = np.concatenate(tiles, axis=1)
        cv2.putText(row, f"frame {idx}", (8, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def write_error_maps(model_maps: dict[str, np.ndarray], out_dir: Path) -> dict[str, np.ndarray]:
    error_dir = out_dir / "temporal_error_maps"
    error_dir.mkdir(parents=True, exist_ok=True)
    errors_by_model = {}
    for name, stack in model_maps.items():
        model_dir = error_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        errors = temporal_error_maps(stack)
        errors_by_model[name] = errors
        np.save(model_dir / "temporal_error.npy", errors)
        for idx, error in enumerate(errors):
            cv2.imwrite(str(model_dir / f"temporal_error_{idx:03d}.png"), colorize_error(error))
    return errors_by_model


def make_depth_error_montage(left_paths, model_maps: dict[str, np.ndarray], errors_by_model: dict[str, np.ndarray], out_path: Path) -> None:
    rows = []
    frames = len(left_paths)
    model_names = list(model_maps.keys())
    for idx in range(frames):
        left = cv2.cvtColor(read_rgb(left_paths[idx]), cv2.COLOR_RGB2BGR)
        left = cv2.resize(left, (240, 144), interpolation=cv2.INTER_AREA)
        cv2.putText(left, f"frame {idx}", (8, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles = [left]
        for name in model_names:
            depth = cv2.resize(colorize_norm(model_maps[name][idx]), (240, 144), interpolation=cv2.INTER_AREA)
            error = cv2.resize(colorize_error(errors_by_model[name][idx]), (240, 144), interpolation=cv2.INTER_AREA)
            cv2.putText(depth, f"{name} depth", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(error, f"{name} temporal error", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.extend([depth, error])
        rows.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def write_csv(path: Path, summary: dict) -> None:
    keys = ["model", "adjacent_delta_mean", "adjacent_delta_p95_mean", "frame_mean_std", "frame_p95_std"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for model, metrics in summary.items():
            row = {"model": model}
            row.update({k: metrics[k] for k in keys[1:]})
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("/home/pampaj/Desktop/stereo/results/video_stereo_repo_scouting/smoke_inputs/scared_rect5"))
    parser.add_argument("--stereoanyvideo_disp", type=Path, default=Path("/home/pampaj/Desktop/stereo/results/stereoanyvideo_scared_smoke/images/disparity.npy"))
    parser.add_argument("--fastfoundation_pattern", default="/home/pampaj/Desktop/stereo/Fast-FoundationStereo/output_scared_rect_kf{idx}")
    parser.add_argument("--fastfoundation_stack", type=Path, default=None)
    parser.add_argument("--skip_fastfoundation", action="store_true")
    parser.add_argument("--s2m2_src", type=Path, default=Path("/home/pampaj/Desktop/stereo/s2m2/src"))
    parser.add_argument("--s2m2_weights", type=Path, default=Path("/home/pampaj/Desktop/stereo/s2m2/weights/pretrain_weights"))
    parser.add_argument("--s2m2_model_type", default="S", choices=["S", "M", "L", "XL"])
    parser.add_argument("--s2m2_refine_iter", type=int, default=3)
    parser.add_argument("--s2m2_max_width", type=int, default=640)
    parser.add_argument("--target_shape", default="384x640")
    parser.add_argument("--out_dir", type=Path, default=Path("/home/pampaj/Desktop/stereo/results/temporal_context_comparison_scared5"))
    args = parser.parse_args()

    h, w = [int(x) for x in args.target_shape.lower().split("x")]
    target_shape = (h, w)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")

    left_paths = sorted((args.input_dir / "left").glob("*.png"))
    right_paths = sorted((args.input_dir / "right").glob("*.png"))
    if len(left_paths) != len(right_paths) or not left_paths:
        raise RuntimeError("Expected matching left/right sequence")

    sav = load_stereoanyvideo_sequence(args.stereoanyvideo_disp, target_shape)
    s2m2 = run_s2m2_sequence(args, left_paths, right_paths, target_shape, args.out_dir)

    model_maps = {
        "stereoanyvideo": sav,
        "s2m2": s2m2,
    }
    if not args.skip_fastfoundation:
        if args.fastfoundation_stack is not None:
            model_maps["fast_foundation"] = load_fastfoundation_stack(args.fastfoundation_stack, target_shape)
        else:
            model_maps["fast_foundation"] = load_fastfoundation_sequence(args.fastfoundation_pattern, len(left_paths), target_shape)
    summary = {name: temporal_metrics(stack) for name, stack in model_maps.items()}
    (args.out_dir / "temporal_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.out_dir / "temporal_metrics.csv", summary)
    make_montage(left_paths, model_maps, args.out_dir / "temporal_context_montage.png")
    errors_by_model = write_error_maps(model_maps, args.out_dir)
    make_depth_error_montage(left_paths, model_maps, errors_by_model, args.out_dir / "temporal_context_depth_error_montage.png")

    report = [
        f"# SCARED {len(left_paths)}-Frame Temporal Context Comparison",
        "",
        "Models: " + ", ".join(model_maps.keys()) + ".",
        "",
        "Metric note: no optical-flow compensation or GT is used here. Adjacent delta is a quick flicker proxy on per-frame robust-normalized maps, so lower generally means smoother but can also mean over-smoothing.",
        "",
        "| model | adjacent_delta_mean | adjacent_delta_p95_mean | frame_mean_std | frame_p95_std |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in summary.items():
        report.append(
            f"| {name} | {metrics['adjacent_delta_mean']:.6f} | {metrics['adjacent_delta_p95_mean']:.6f} | {metrics['frame_mean_std']:.6f} | {metrics['frame_p95_std']:.6f} |"
        )
    report.extend([
        "",
        "See `temporal_context_montage.png` for the depth-map visual comparison.",
        "See `temporal_context_depth_error_montage.png` for depth maps plus temporal error maps.",
        "Per-frame temporal error maps are in `temporal_error_maps/<model>/`.",
    ])
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
