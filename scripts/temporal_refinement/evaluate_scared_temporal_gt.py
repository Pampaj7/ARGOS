#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.temporal_refinement.lib.models import ConvGRURefiner, TinyUNetRefiner
from scripts.temporal_refinement.lib.training import colorize


S2M2_REPO = Path("../../external/frame_stereo_repos/s2m2")
S2M2_SRC = S2M2_REPO / "src"
S2M2_WEIGHTS = S2M2_REPO / "weights/pretrain_weights"
SAV_REPO = Path("../../external/video_stereo_repos/stereoanyvideo")
SAV_CKPT = SAV_REPO / "checkpoints/StereoAnyVideo_MIX.pth"


S2M2_CONFIGS = {
    "S": {"feature_channels": 128, "num_transformer": 1, "weight": "CH128NTR1.pth"},
    "L": {"feature_channels": 256, "num_transformer": 3, "weight": "CH256NTR3.pth"},
    "XL": {"feature_channels": 384, "num_transformer": 3, "weight": "CH384NTR3.pth"},
}


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_metadata(sequence_root: Path, max_frames: int = 0) -> list[dict]:
    with (sequence_root / "metadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    if max_frames:
        rows = rows[:max_frames]
    frames = []
    for row in rows:
        fid = row.get("frame_id") or row.get("id")
        frame = {
            "id": fid,
            "left_path": path_from_manifest(sequence_root, row.get("left_path"), sequence_root / "left" / f"{fid}.png"),
            "right_path": path_from_manifest(sequence_root, row.get("right_path"), sequence_root / "right" / f"{fid}.png"),
            "gt_disp_path": path_from_manifest(sequence_root, row.get("disparity_float32_path"), sequence_root / "gt" / "disp" / f"{fid}.npy"),
            "gt_depth_path": path_from_manifest(sequence_root, row.get("depth_float32_path"), sequence_root / "gt" / "depth" / f"{fid}.npy"),
            "valid_mask_path": path_from_manifest(sequence_root, row.get("valid_mask_path"), sequence_root / "gt" / "valid_mask" / f"{fid}.png"),
            "calib_path": path_from_manifest(sequence_root, row.get("calibration_path"), sequence_root / "calibration" / f"{fid}.json"),
        }
        calib = json.loads(frame["calib_path"].read_text())
        frame["fx"], frame["baseline_mm"] = calibration_fx_baseline(calib)
        valid = read_valid_mask(frame["valid_mask_path"])
        frame["valid_ratio"] = float(valid.mean())
        frames.append(frame)
    return frames


def path_from_manifest(sequence_root: Path, value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    rooted = ROOT / path
    if rooted.exists():
        return rooted
    return sequence_root / path


def calibration_fx_baseline(calib: dict) -> tuple[float, float]:
    if "fx" in calib and "baseline_mm" in calib:
        return float(calib["fx"]), float(calib["baseline_mm"])
    p1 = np.array(calib["P1"]["data"], dtype=np.float64).reshape(calib["P1"]["rows"], calib["P1"]["cols"])
    p2 = np.array(calib["P2"]["data"], dtype=np.float64).reshape(calib["P2"]["rows"], calib["P2"]["cols"])
    fx = float(p1[0, 0])
    baseline = float(abs(p2[0, 3] / p2[0, 0]))
    return fx, baseline


def read_valid_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(bool)
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0


def load_frame_payload(frame: dict) -> dict:
    valid = read_valid_mask(frame["valid_mask_path"])
    return {
        "left": read_rgb(frame["left_path"]),
        "right": read_rgb(frame["right_path"]),
        "gt_disp": np.load(frame["gt_disp_path"]).astype(np.float32),
        "gt_depth": np.load(frame["gt_depth_path"]).astype(np.float32),
        "valid": valid,
        "fx": frame["fx"],
        "baseline_mm": frame["baseline_mm"],
    }


def save_disp(path: Path, disp: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, disp.astype(np.float32))


def load_or_none(path: Path) -> np.ndarray | None:
    if path.exists():
        return np.load(path).astype(np.float32)
    return None


def build_s2m2(variant: str, device: torch.device):
    sys.path.insert(0, str(S2M2_SRC))
    from s2m2.core.model.s2m2 import S2M2

    cfg = S2M2_CONFIGS[variant]
    model = S2M2(
        feature_channels=cfg["feature_channels"],
        dim_expansion=1,
        num_transformer=cfg["num_transformer"],
        use_positivity=True,
        refine_iter=3,
    )
    ckpt = torch.load(S2M2_WEIGHTS / cfg["weight"], map_location="cpu", weights_only=True)
    model.my_load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def infer_s2m2_frame(model, left: np.ndarray, right: np.ndarray, width: int, device: torch.device) -> tuple[np.ndarray, float, float]:
    sys.path.insert(0, str(S2M2_SRC))
    from s2m2.core.utils.image_utils import image_crop, image_pad

    orig_h, orig_w = left.shape[:2]
    if width <= 0:
        left_in, right_in, scale_x = left, right, 1.0
    else:
        scale_x = width / float(orig_w)
        new_h = int(round(orig_h * scale_x))
        left_in = cv2.resize(left, (width, new_h), interpolation=cv2.INTER_LINEAR)
        right_in = cv2.resize(right, (width, new_h), interpolation=cv2.INTER_LINEAR)
    left_t = torch.from_numpy(left_in).permute(2, 0, 1).unsqueeze(0).float().to(device)
    right_t = torch.from_numpy(right_in).permute(2, 0, 1).unsqueeze(0).float().to(device)
    h, w = left_t.shape[-2:]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        pred, _occ, _conf = model(image_pad(left_t, 32), image_pad(right_t, 32))
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    pred = image_crop(pred, (h, w)).squeeze().float().cpu().numpy()
    if scale_x != 1.0:
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale_x
    return np.clip(pred.astype(np.float32), 0, None), runtime_ms, scale_x


def run_s2m2(frames: list[dict], out_dir: Path, variant: str, width: int, device: torch.device) -> dict:
    name = f"S2M2-{variant}@{width if width else 'full'}"
    pred_dir = out_dir / "predictions" / safe_name(name)
    pred_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in frames if not (pred_dir / f"{f['id']}.npy").exists()]
    runtimes = []
    peak = 0.0
    if missing:
        model = build_s2m2(variant, device)
        for frame in frames:
            dst = pred_dir / f"{frame['id']}.npy"
            if dst.exists():
                continue
            payload = load_frame_payload(frame)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            pred, runtime_ms, _scale_x = infer_s2m2_frame(model, payload["left"], payload["right"], width, device)
            save_disp(dst, pred)
            runtimes.append(runtime_ms)
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated() / (1024**2))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metadata = {
        "method": name,
        "kind": "frame_stereo",
        "variant": variant,
        "input_resolution": str(width if width else "full"),
        "frames": len(frames),
        "runtime_ms_values": runtimes,
        "avg_runtime_ms": float(np.mean(runtimes)) if runtimes else np.nan,
        "median_runtime_ms": float(np.median(runtimes)) if runtimes else np.nan,
        "peak_vram_mb": peak,
        "coordinate_system": "original image disparity coordinates",
    }
    (pred_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def build_sav(device: torch.device):
    cwd = Path.cwd()
    os.chdir(SAV_REPO)
    sys.path.insert(0, str(SAV_REPO.parent))
    sys.path.insert(0, str(SAV_REPO))
    from stereoanyvideo.models.core.stereoanyvideo import StereoAnyVideo

    model = StereoAnyVideo(mixed_precision=False)
    state = torch.load(SAV_CKPT, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    if "state_dict" in state:
        state = state["state_dict"]
    if state and next(iter(state)).startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    os.chdir(cwd)
    return model.to(device).eval()


@torch.no_grad()
def infer_sav_chunk(model, frames: list[dict], resize_hw: tuple[int, int], iters: int, device: torch.device) -> tuple[list[np.ndarray], float, float]:
    h, w = resize_hw
    lefts, rights = [], []
    first_payload = load_frame_payload(frames[0])
    orig_h, orig_w = first_payload["left"].shape[:2]
    for idx, frame in enumerate(frames):
        payload = first_payload if idx == 0 else load_frame_payload(frame)
        left = torch.from_numpy(payload["left"]).permute(2, 0, 1).float().to(device)
        right = torch.from_numpy(payload["right"]).permute(2, 0, 1).float().to(device)
        lefts.append(F.interpolate(left[None], size=(h, w), mode="bilinear", align_corners=True)[0])
        rights.append(F.interpolate(right[None], size=(h, w), mode="bilinear", align_corners=True)[0])
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    stereo_video = torch.stack([torch.stack(lefts, 0), torch.stack(rights, 0)], dim=1)
    raw = model.forward(stereo_video[:, 0][None], stereo_video[:, 1][None], iters=iters, test_mode=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    peak = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    if raw.shape[0] == len(frames):
        disp = raw[:, 0, :1].abs()
    else:
        disp = raw[0, :, :1].abs()
    disp_np = disp.squeeze(1).float().cpu().numpy().astype(np.float32)
    scale_x = w / float(orig_w)
    preds = [cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale_x for d in disp_np]
    return preds, runtime_ms / max(len(frames), 1), peak


def run_sav(frames: list[dict], out_dir: Path, device: torch.device, chunk_size: int, overlap: int) -> dict:
    name = "StereoAnyVideo@384x640"
    pred_dir = out_dir / "predictions" / safe_name(name)
    pred_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in frames if not (pred_dir / f"{f['id']}.npy").exists()]
    runtimes, chunk_rows = [], []
    peak = 0.0
    if missing:
        model = build_sav(device)
        cursor = 0
        chunk_idx = 0
        written = set()
        while cursor < len(frames):
            chunk = frames[cursor : min(cursor + chunk_size, len(frames))]
            preds, per_frame_ms, p = infer_sav_chunk(model, chunk, (384, 640), 6, device)
            peak = max(peak, p)
            runtimes.extend([per_frame_ms] * len(chunk))
            for frame, pred in zip(chunk, preds):
                # First write wins in overlap regions, preserving earlier temporal context.
                if frame["id"] not in written and not (pred_dir / f"{frame['id']}.npy").exists():
                    save_disp(pred_dir / f"{frame['id']}.npy", np.clip(pred, 0, None))
                    written.add(frame["id"])
            chunk_rows.append({"chunk": chunk_idx, "start": chunk[0]["id"], "end": chunk[-1]["id"], "frames": len(chunk), "runtime_ms_per_frame": per_frame_ms})
            if cursor + chunk_size >= len(frames):
                break
            cursor += max(1, chunk_size - overlap)
            chunk_idx += 1
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metadata = {
        "method": name,
        "kind": "video_stereo",
        "input_resolution": "384x640",
        "frames": len(frames),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": chunk_rows,
        "avg_runtime_ms": float(np.mean(runtimes)) if runtimes else np.nan,
        "median_runtime_ms": float(np.median(runtimes)) if runtimes else np.nan,
        "peak_vram_mb": peak,
        "coordinate_system": "original image disparity coordinates",
    }
    (pred_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def safe_name(name: str) -> str:
    return name.replace("@", "_").replace("/", "_").replace(" ", "_")


def load_predictions(out_dir: Path, method: str, frames: list[dict]) -> list[np.ndarray]:
    pred_dir = out_dir / "predictions" / safe_name(method)
    return [np.load(pred_dir / f"{frame['id']}.npy").astype(np.float32) for frame in frames]


def load_checkpoint(path: Path, model_type: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    base = int(args.get("base_channels", 16))
    clamp = float(args.get("residual_clamp_px", 2.0))
    if model_type == "tiny_unet":
        model = TinyUNetRefiner(in_channels=8, base_channels=base, residual_clamp_px=clamp)
    elif model_type == "convgru":
        hidden = int(args.get("hidden_channels", 64))
        model = ConvGRURefiner(in_channels=4, base_channels=base, hidden_channels=hidden, residual_clamp_px=clamp)
    else:
        raise ValueError(model_type)
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval(), ck


@torch.no_grad()
def run_tiny_unet(
    frames: list[dict],
    out_dir: Path,
    checkpoint: Path,
    raw_method: str,
    device: torch.device,
    disp_norm: float,
) -> dict:
    name = f"TinyUNet-L736@{checkpoint.parent.parent.name}:{checkpoint.stem}"
    pred_dir = out_dir / "predictions" / safe_name(name)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model, _ck = load_checkpoint(checkpoint, "tiny_unet", device)
    raw = load_predictions(out_dir, raw_method, frames)
    runtimes, residuals = [], []
    peak = 0.0
    for i, frame in enumerate(frames):
        dst = pred_dir / f"{frame['id']}.npy"
        if dst.exists():
            continue
        if i < 2 or i + 2 >= len(frames):
            save_disp(dst, raw[i])
            continue
        rgb = read_rgb(frame["left_path"]).astype(np.float32) / 255.0
        window = np.stack(raw[i - 2 : i + 3], axis=0)
        x = np.concatenate([rgb.transpose(2, 0, 1), window / disp_norm], axis=0)
        x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)
        center = torch.from_numpy(raw[i]).unsqueeze(0).unsqueeze(0).float().to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            delta = model(x_t)
            refined = torch.clamp(center + delta, min=0.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        runtimes.append((time.perf_counter() - t0) * 1000.0)
        pred = refined[0, 0].float().cpu().numpy()
        save_disp(dst, pred)
        residuals.append(pred - raw[i])
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / (1024**2))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metadata = {
        "method": name,
        "kind": "temporal_refiner",
        "checkpoint": str(checkpoint),
        "causal": False,
        "avg_runtime_ms": float(np.mean(runtimes)) if runtimes else np.nan,
        "median_runtime_ms": float(np.median(runtimes)) if runtimes else np.nan,
        "peak_vram_mb": peak,
        "residual_abs_mean": float(np.mean([np.mean(np.abs(x)) for x in residuals])) if residuals else 0.0,
    }
    (pred_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


@torch.no_grad()
def run_convgru(
    frames: list[dict],
    out_dir: Path,
    checkpoint: Path,
    raw_method: str,
    device: torch.device,
    disp_norm: float,
) -> dict:
    name = f"ConvGRU-L736@{checkpoint.parent.parent.name}:{checkpoint.stem}"
    pred_dir = out_dir / "predictions" / safe_name(name)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model, _ck = load_checkpoint(checkpoint, "convgru", device)
    raw = load_predictions(out_dir, raw_method, frames)
    hidden = None
    runtimes, residuals = [], []
    peak = 0.0
    for i, frame in enumerate(frames):
        dst = pred_dir / f"{frame['id']}.npy"
        if dst.exists():
            continue
        rgb = read_rgb(frame["left_path"]).astype(np.float32) / 255.0
        x = np.concatenate([rgb.transpose(2, 0, 1), raw[i][None] / disp_norm], axis=0)
        x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)
        center = torch.from_numpy(raw[i]).unsqueeze(0).unsqueeze(0).float().to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            delta, hidden = model(x_t, hidden)
            refined = torch.clamp(center + delta, min=0.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        runtimes.append((time.perf_counter() - t0) * 1000.0)
        pred = refined[0, 0].float().cpu().numpy()
        save_disp(dst, pred)
        residuals.append(pred - raw[i])
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / (1024**2))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metadata = {
        "method": name,
        "kind": "temporal_refiner",
        "checkpoint": str(checkpoint),
        "causal": True,
        "avg_runtime_ms": float(np.mean(runtimes)) if runtimes else np.nan,
        "median_runtime_ms": float(np.median(runtimes)) if runtimes else np.nan,
        "peak_vram_mb": peak,
        "residual_abs_mean": float(np.mean([np.mean(np.abs(x)) for x in residuals])) if residuals else 0.0,
    }
    (pred_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def disp_to_depth(pred: np.ndarray, frame: dict) -> np.ndarray:
    return frame["fx"] * frame["baseline_mm"] / np.maximum(pred.astype(np.float32), 1e-6)


def method_metrics(method: str, preds: list[np.ndarray], frames: list[dict], min_valid_ratio: float, metadata: dict) -> tuple[dict, list[dict]]:
    per_frame = []
    disp_stack = []
    valid_stack = []
    for pred, frame in zip(preds, frames):
        payload = load_frame_payload(frame)
        valid = payload["valid"] & (payload["gt_disp"] > 0) & (payload["gt_depth"] > 0) & np.isfinite(pred) & (pred > 0.1)
        pred_depth = disp_to_depth(pred, frame)
        row = {
            "method": method,
            "frame_id": frame["id"],
            "valid_ratio": float(valid.mean()),
            "included_for_geometry": bool(valid.mean() >= min_valid_ratio),
            "valid_disp_mae": np.nan,
            "valid_disp_rmse": np.nan,
            "bad_1px": np.nan,
            "bad_2px": np.nan,
            "bad_3px": np.nan,
            "valid_depth_mae": np.nan,
            "valid_depth_median": np.nan,
            "valid_depth_rmse": np.nan,
            "bad_1mm": np.nan,
            "bad_2mm": np.nan,
            "bad_5mm": np.nan,
            "pred_disp_le_0_1_ratio": float(((payload["valid"]) & (pred <= 0.1)).sum() / max(int(payload["valid"].sum()), 1)),
            "pred_disp_le_0_5_ratio": float(((payload["valid"]) & (pred <= 0.5)).sum() / max(int(payload["valid"].sum()), 1)),
        }
        if valid.any():
            disp_err = np.abs(pred[valid] - payload["gt_disp"][valid])
            depth_err = np.abs(pred_depth[valid] - payload["gt_depth"][valid])
            row.update(
                {
                    "valid_disp_mae": float(disp_err.mean()),
                    "valid_disp_rmse": float(np.sqrt((disp_err**2).mean())),
                    "bad_1px": float((disp_err > 1).mean() * 100.0),
                    "bad_2px": float((disp_err > 2).mean() * 100.0),
                    "bad_3px": float((disp_err > 3).mean() * 100.0),
                    "valid_depth_mae": float(depth_err.mean()),
                    "valid_depth_median": float(np.median(depth_err)),
                    "valid_depth_rmse": float(np.sqrt((depth_err**2).mean())),
                    "bad_1mm": float((depth_err > 1).mean() * 100.0),
                    "bad_2mm": float((depth_err > 2).mean() * 100.0),
                    "bad_5mm": float((depth_err > 5).mean() * 100.0),
                }
            )
        per_frame.append(row)
        disp_stack.append(pred)
        valid_stack.append(valid)

    included = [r for r in per_frame if r["included_for_geometry"]]
    temporal = []
    temporal_depth = []
    temporal_error_variation = []
    for i in range(1, len(preds)):
        both = valid_stack[i] & valid_stack[i - 1]
        if not both.any():
            continue
        temporal.append(float(np.abs(preds[i] - preds[i - 1])[both].mean()))
        temporal_depth.append(float(np.abs(disp_to_depth(preds[i], frames[i]) - disp_to_depth(preds[i - 1], frames[i - 1]))[both].mean()))
        payload_i = load_frame_payload(frames[i])
        payload_p = load_frame_payload(frames[i - 1])
        err_i = np.abs(preds[i] - payload_i["gt_disp"])
        err_p = np.abs(preds[i - 1] - payload_p["gt_disp"])
        temporal_error_variation.append(float(np.abs(err_i - err_p)[both].mean()))
    stack = np.stack(disp_stack, axis=0)
    valid_all = np.stack(valid_stack, axis=0)
    any_valid = valid_all.any(axis=0)
    temporal_std = float(np.nanmean(np.where(any_valid, np.nanstd(np.where(valid_all, stack, np.nan), axis=0), np.nan)))

    def mean_key(key: str) -> float:
        vals = [r[key] for r in included if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else np.nan

    summary = {
        "method": method,
        "training_or_checkpoint": metadata.get("checkpoint", metadata.get("variant", "")),
        "input_res": metadata.get("input_resolution", ""),
        "frames": len(frames),
        "frames_with_gt_used": len(included),
        "min_valid_ratio": min_valid_ratio,
        "Depth MAE ↓": mean_key("valid_depth_mae"),
        "Bad-2 mm ↓": mean_key("bad_2mm"),
        "Disp. MAE ↓": mean_key("valid_disp_mae"),
        "Disp. RMSE ↓": mean_key("valid_disp_rmse"),
        "Bad-3 px ↓": mean_key("bad_3px"),
        "Temporal diff ↓": float(np.nanmean(temporal)) if temporal else np.nan,
        "Temporal depth diff ↓": float(np.nanmean(temporal_depth)) if temporal_depth else np.nan,
        "Temporal error variation ↓": float(np.nanmean(temporal_error_variation)) if temporal_error_variation else np.nan,
        "Temporal std ↓": temporal_std,
        "Runtime ↓": metadata.get("avg_runtime_ms", np.nan),
        "VRAM ↓": metadata.get("peak_vram_mb", np.nan),
        "causal": metadata.get("causal", ""),
        "kind": metadata.get("kind", ""),
        "evidence_source": str((Path("results/03_temporal_refinement/evaluation/gt_temporal_test_dataset_9_keyframe_3") / "predictions" / safe_name(method) / "metadata.json")),
    }
    return summary, per_frame


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_montages(out_dir: Path, frames: list[dict], methods: list[str], selected: list[int]) -> None:
    qdir = out_dir / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    for idx in selected:
        if idx < 0 or idx >= len(frames):
            continue
        frame = frames[idx]
        rgb = read_rgb(frame["left_path"])
        preds = {m: load_predictions(out_dir, m, [frame])[0] for m in methods}
        gt = np.load(frame["gt_disp_path"]).astype(np.float32)
        vmax = float(np.nanpercentile(np.concatenate([gt.ravel()] + [p.ravel() for p in preds.values()]), 99))
        tiles = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colorize(gt, vmax)]
        labels = ["RGB", "GT disp"]
        for method, pred in preds.items():
            tiles.append(colorize(pred, vmax))
            labels.append(method[:20])
            tiles.append(colorize(np.abs(pred - gt), 10.0, cv2.COLORMAP_MAGMA))
            labels.append("abs err")
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (220, 176), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(tile, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
        cv2.imwrite(str(qdir / f"frame_{frame['id']}.png"), np.concatenate(small, axis=1))


def write_report(out_dir: Path, summaries: list[dict], sequence_root: Path, methods: list[str]) -> None:
    table_cols = ["method", "training_or_checkpoint", "input_res", "frames_with_gt_used", "Depth MAE ↓", "Bad-2 mm ↓", "Disp. MAE ↓", "Temporal diff ↓", "Runtime ↓", "VRAM ↓", "causal"]
    lines = [
        "# Temporal Evaluation With SCARED GT",
        "",
        f"Sequence: `{sequence_root}`",
        "",
        "This table is regenerated on rectified frames with GT attached. The old long temporal cache is not scored against this GT because its frames are not pixel-aligned with the newly converted rectified SCARED frames.",
        "",
        "| " + " | ".join(table_cols) + " |",
        "| " + " | ".join(["---"] * len(table_cols)) + " |",
    ]
    for row in summaries:
        vals = []
        for col in table_cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append("" if not np.isfinite(val) else f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- GT source: raw `dataset_9.zip`, block `keyframe_3`, converted into rectified left/right plus depth/disparity/mask.",
            "- Geometry metrics use only frames whose GT valid-pixel ratio passes the configured threshold.",
            "- Disparity predictions are saved in original image coordinates before metric computation.",
            "- StereoAnyVideo is run in chunks; it remains a video-native teacher/baseline, but chunking can slightly affect long-range temporal context.",
            "- Temporal smoothness is not geometric correctness; here we report both GT errors and temporal differences.",
            "",
            "## Methods Included",
            "",
        ]
    )
    lines.extend([f"- `{m}`" for m in methods])
    (out_dir / "temporal_evaluation_gt.md").write_text("\n".join(lines) + "\n")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence-root", type=Path, default=ROOT / "dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/03_temporal_refinement/evaluation/gt_temporal_test_dataset_9_keyframe_3")
    p.add_argument("--max-frames", type=int, default=130)
    p.add_argument("--min-valid-ratio", type=float, default=0.2)
    p.add_argument("--sav-chunk-size", type=int, default=32)
    p.add_argument("--sav-overlap", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-refiners", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    frames = read_metadata(args.sequence_root, args.max_frames)
    print(f"frames={len(frames)} device={device} out={args.out_dir}", flush=True)

    method_meta = []
    method_meta.append(run_s2m2(frames, args.out_dir, "L", 736, device))
    method_meta.append(run_s2m2(frames, args.out_dir, "S", 512, device))
    method_meta.append(run_sav(frames, args.out_dir, device, args.sav_chunk_size, args.sav_overlap))

    if not args.skip_refiners:
        ckpts = [
            ("convgru", ROOT / "results/03_temporal_refinement/training/convgru/temporal_refinement_train_convgru_l736_v2_scheduled/checkpoints/epoch_0030.pt"),
            ("convgru", ROOT / "results/03_temporal_refinement/training/convgru/temporal_refinement_train_convgru_l736_v2_scheduled/checkpoints/epoch_0040.pt"),
            ("convgru", ROOT / "results/03_temporal_refinement/training/convgru/temporal_refinement_train_convgru_l736_v2_scheduled/checkpoints/epoch_0050.pt"),
            ("convgru", ROOT / "results/03_temporal_refinement/training/convgru/temporal_refinement_train_convgru_l736_v2_scheduled/checkpoints/latest.pt"),
            ("tiny_unet", ROOT / "results/03_temporal_refinement/training/unet/temporal_refinement_train_unet_s2m2l736_fastcache_v2_conservative/checkpoints/epoch_0100.pt"),
        ]
        for model_type, ckpt in ckpts:
            if not ckpt.exists():
                continue
            if model_type == "convgru":
                method_meta.append(run_convgru(frames, args.out_dir, ckpt, "S2M2-L@736", device, 128.0))
            else:
                method_meta.append(run_tiny_unet(frames, args.out_dir, ckpt, "S2M2-L@736", device, 128.0))

    summaries = []
    per_frame_all = []
    methods = [m["method"] for m in method_meta]
    for meta in method_meta:
        preds = load_predictions(args.out_dir, meta["method"], frames)
        summary, per_frame = method_metrics(meta["method"], preds, frames, args.min_valid_ratio, meta)
        summaries.append(summary)
        per_frame_all.extend(per_frame)
    write_csv(args.out_dir / "temporal_evaluation_gt.csv", summaries)
    write_csv(args.out_dir / "per_frame_metrics.csv", per_frame_all)
    (args.out_dir / "temporal_evaluation_gt.json").write_text(json.dumps({"args": vars(args) | {"sequence_root": str(args.sequence_root), "out_dir": str(args.out_dir)}, "summary": summaries}, indent=2) + "\n")
    save_montages(args.out_dir, frames, methods[:3] + [m for m in methods if m.startswith("ConvGRU")][:2], [0, len(frames) // 2, min(len(frames) - 1, 100)])
    write_report(args.out_dir, summaries, args.sequence_root, methods)
    print(json.dumps({"out": str(args.out_dir), "methods": methods, "csv": str(args.out_dir / "temporal_evaluation_gt.csv")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
