#!/usr/bin/env python3
"""Evaluate one external stereo model on the native curated SCARED protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
STEREO_ROOT = ROOT.parent / "stereo"
sys.path.insert(0, str(ROOT / "scripts/scared"))

from benchmark_s2m2_size_tradeoff import collect_samples  # noqa: E402
from eval_metrics import failure_aware_metrics  # noqa: E402


def resolve_data_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def read_rgb_path(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def collect_samples_from_metadata(metadata_csv: Path) -> list[dict]:
    samples = []
    with metadata_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("notes"):
                continue
            left_path = resolve_data_path(row["left_path"])
            right_path = resolve_data_path(row["right_path"])
            depth_path = resolve_data_path(row["depth_float32_path"])
            disp_path = resolve_data_path(row["disparity_float32_path"])
            mask_path = resolve_data_path(row["valid_mask_path"])
            calib_path = resolve_data_path(row["calibration_path"])
            required = [left_path, right_path, depth_path, disp_path, mask_path, calib_path]
            if not all(p.exists() for p in required):
                missing = [str(p) for p in required if not p.exists()]
                raise FileNotFoundError(f"Missing metadata sample files for {row}: {missing}")
            calib = json.loads(calib_path.read_text())
            p1 = np.array(calib["P1"]["data"], dtype=np.float64).reshape(3, 4)
            p2 = np.array(calib["P2"]["data"], dtype=np.float64).reshape(3, 4)
            if row.get("dataset_id") and row.get("keyframe_id"):
                frame = f"{row['dataset_id']}_{row['keyframe_id']}_frame_{int(row['frame_id']):06d}"
            else:
                frame = f"{row.get('sequence_id', 'sequence')}_frame_{int(row['frame_id']):06d}"
            samples.append(
                {
                    "frame": frame,
                    "left": read_rgb_path(left_path),
                    "right": read_rgb_path(right_path),
                    "gt_disp": np.load(disp_path).astype(np.float32),
                    "gt_depth": np.load(depth_path).astype(np.float32),
                    "valid": np.load(mask_path).astype(bool),
                    "fx": float(p1[0, 0]),
                    "baseline_mm": float(abs(p2[0, 3] / p2[0, 0])),
                }
            )
    if not samples:
        raise RuntimeError(f"No usable SCARED samples found in {metadata_csv}")
    return samples


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    skip = {"method", "checkpoint", "input_resolution", "frame"}
    numeric = [k for k, v in rows[0].items() if k not in skip and isinstance(v, (int, float))]
    return {k: float(np.mean([r[k] for r in rows])) for k in numeric}


def metric_row(pred: np.ndarray, sample: dict) -> dict:
    pred = np.clip(pred.astype(np.float32), 0, None)
    pred_depth = sample["fx"] * sample["baseline_mm"] / np.maximum(pred, 1e-6)
    raw_mask = (
        sample["valid"]
        & np.isfinite(pred)
        & np.isfinite(pred_depth)
        & (sample["gt_disp"] > 0)
        & (sample["gt_depth"] > 0)
    )
    return failure_aware_metrics(
        pred,
        pred_depth,
        sample["gt_disp"],
        sample["gt_depth"],
        sample["valid"],
        raw_mask,
    )


def image_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)


class SimplePadder:
    def __init__(self, shape, divis_by: int = 32):
        ht, wd = shape[-2:]
        pad_ht = (((ht // divis_by) + 1) * divis_by - ht) % divis_by
        pad_wd = (((wd // divis_by) + 1) * divis_by - wd) % divis_by
        self.pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad_tensors(self, *inputs):
        return [F.pad(x, self.pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        left, right, top, bottom = self.pad
        return x[..., top : ht - bottom, left : wd - right]


def eval_crestereo(samples: list[dict], args, device: torch.device):
    repo = STEREO_ROOT / "stereo_matching_crestereo"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from stereo_matching_crestereo import CrestereoMatching

    matcher = CrestereoMatching({"max_disp": 256}).to(device).eval()

    def predict(left, right):
        t0 = time.perf_counter()
        pred = matcher(left, right)["disparity"].astype(np.float32)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return pred, (time.perf_counter() - t0) * 1000.0

    return "CREStereo", "local checkpoint", "native curated SCARED", predict


def eval_raft(samples: list[dict], args, device: torch.device):
    repo = STEREO_ROOT / "RAFT-Stereo"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "core"))
    from raft_stereo import RAFTStereo, autocast
    from utils.utils import InputPadder

    cfg = SimpleNamespace(
        restore_ckpt=str(repo / "models/raftstereo-middlebury.pth"),
        mixed_precision=False,
        valid_iters=32,
        hidden_dims=[128] * 3,
        corr_implementation="reg",
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
    )
    model = torch.nn.DataParallel(RAFTStereo(cfg), device_ids=[0])
    state = torch.load(cfg.restore_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.module.to(device).eval()

    @torch.no_grad()
    def predict(left, right):
        image1 = image_to_tensor(left, device)
        image2 = image_to_tensor(right, device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast(enabled=cfg.mixed_precision):
            _, flow_up = model(image1, image2, iters=cfg.valid_iters, test_mode=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pred = -padder.unpad(flow_up).squeeze().float().cpu().numpy()
        return pred.astype(np.float32), (time.perf_counter() - t0) * 1000.0

    return "RAFT-Stereo Middlebury", "raftstereo-middlebury.pth", "native curated SCARED", predict


def eval_defom(samples: list[dict], args, device: torch.device):
    repo = STEREO_ROOT / "DEFOM-Stereo"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from core.defom_stereo import DEFOMStereo

    cfg = SimpleNamespace(
        restore_ckpt=str(repo / "checkpoints/defomstereo_vitl_eth3d.pth"),
        mixed_precision=False,
        valid_iters=16,
        scale_iters=4,
        dinov2_encoder="vitl",
        idepth_scale=0.5,
        hidden_dims=[128] * 3,
        corr_implementation="reg",
        shared_backbone=False,
        corr_levels=2,
        corr_radius=4,
        scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        scale_corr_radius=2,
        n_downsample=2,
        context_norm="batch",
        n_gru_layers=3,
    )
    model = DEFOMStereo(cfg)
    checkpoint = torch.load(cfg.restore_ckpt, map_location="cpu")
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    @torch.no_grad()
    def predict(left, right):
        image1 = image_to_tensor(left, device)
        image2 = image_to_tensor(right, device)
        padder = SimplePadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad_tensors(image1, image2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(image1, image2, iters=cfg.valid_iters, scale_iters=cfg.scale_iters, test_mode=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pred = padder.unpad(pred).squeeze().float().cpu().numpy()
        return pred.astype(np.float32), (time.perf_counter() - t0) * 1000.0

    return "DEFOM-Stereo ViT-L ETH3D", "defomstereo_vitl_eth3d.pth", "native curated SCARED", predict


def eval_monster(samples: list[dict], args, device: torch.device, realtime: bool = False):
    repo = STEREO_ROOT / "MonSter-plusplus" / ("RT-MonSter++" if realtime else "MonSter++")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "core"))
    from monster import Monster
    from utils.utils import InputPadder

    if realtime:
        cfg = SimpleNamespace(
            restore_ckpt=str(repo / "checkpoints/Zero_shot.pth"),
            valid_iters=4,
            encoder="vits",
            hidden_dims=[32, 64, 96],
            corr_implementation="reg",
            shared_backbone=False,
            corr_levels=2,
            corr_radius=[2, 2, 4],
            n_downsample=2,
            slow_fast_gru=False,
            n_gru_layers=3,
            max_disp=192,
        )
        method = "RT-MonSter++ zero-shot"
        checkpoint_name = "Zero_shot.pth"
    else:
        cfg = SimpleNamespace(
            restore_ckpt=str(repo / "checkpoints/Mix_all_large.pth"),
            valid_iters=16,
            encoder="vitl",
            hidden_dims=[128] * 3,
            corr_implementation="reg",
            shared_backbone=False,
            corr_levels=2,
            corr_radius=4,
            n_downsample=2,
            slow_fast_gru=False,
            n_gru_layers=3,
            max_disp=416,
        )
        method = "MonSter++ MixAll"
        checkpoint_name = "Mix_all_large.pth"

    model = torch.nn.DataParallel(Monster(cfg), device_ids=[0])
    checkpoint = torch.load(cfg.restore_ckpt, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]
    state = {key if key.startswith("module.") else f"module.{key}": value for key, value in checkpoint.items()}
    model.load_state_dict(state, strict=True)
    model = model.module.to(device).eval()

    @torch.no_grad()
    def predict(left, right):
        image1 = image_to_tensor(left, device)
        image2 = image_to_tensor(right, device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(image1, image2, iters=cfg.valid_iters, test_mode=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pred = padder.unpad(pred).squeeze().float().cpu().numpy()
        return pred.astype(np.float32), (time.perf_counter() - t0) * 1000.0

    return method, checkpoint_name, "native curated SCARED", predict


def eval_stereoanywhere(samples: list[dict], args, device: torch.device):
    repo = STEREO_ROOT / "stereoanywhere"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from models.depth_anything_v2 import get_depth_anything_v2
    from models.stereoanywhere import StereoAnywhere
    import torch.nn.functional as F

    cfg = SimpleNamespace(
        maxdisp=192,
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
        mixed_precision=False,
        corr_implementation="reg",
    )
    model = torch.nn.DataParallel(StereoAnywhere(cfg)).to(device).eval()
    checkpoint = torch.load(repo / "weights/stereoanywhere_sceneflow.pth", map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    mono_model = get_depth_anything_v2(repo / "weights/depth_anything_v2_vits.pth", encoder="vits").eval().to(device)

    def to_tensor(img):
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    @torch.no_grad()
    def predict(left, right):
        h, w = left.shape[:2]
        left_t = to_tensor(left)
        right_t = to_tensor(right)
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_disps, _ = model(left_t, right_t, left_mono, right_mono, test_mode=True, iters=32)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pred = pred_disps.squeeze().float().detach().cpu().numpy()
        ph, pw = pred.shape[-2:]
        pred = pred[pad[2] : ph - pad[3], pad[0] : pw - pad[1]]
        if np.count_nonzero(pred > 0) < np.count_nonzero((-pred) > 0):
            pred = -pred
        return np.clip(pred, 0, None).astype(np.float32), (time.perf_counter() - t0) * 1000.0

    return "StereoAnywhere", "stereoanywhere_sceneflow + DepthAnything-V2-S", "native curated SCARED", predict


BUILDERS = {
    "crestereo": eval_crestereo,
    "raft_middlebury": eval_raft,
    "defom_vitl_eth3d": eval_defom,
    "monster_mixall": lambda samples, args, device: eval_monster(samples, args, device, realtime=False),
    "rtmonster_zeroshot": lambda samples, args, device: eval_monster(samples, args, device, realtime=True),
    "stereoanywhere": eval_stereoanywhere,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--scared-root", type=Path, default=ROOT / "dataset/SCARED/curated/keyframes_gt_dataset8/dataset_8")
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--input-resolution-label", default="native curated SCARED")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.scared_root = args.scared_root.resolve()
    args.metadata_csv = args.metadata_csv.resolve() if args.metadata_csv else None
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples_from_metadata(args.metadata_csv) if args.metadata_csv else collect_samples(args.scared_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    method, checkpoint, input_resolution, predict = BUILDERS[args.model](samples, args, device)
    input_resolution = args.input_resolution_label or input_resolution

    rows = []
    for sample in samples:
        pred, runtime_ms = predict(sample["left"], sample["right"])
        np.save(args.out_dir / f"{sample['frame']}_disp.npy", pred.astype(np.float32))
        row = {
            "method": method,
            "checkpoint": checkpoint,
            "input_resolution": input_resolution,
            "frame": sample["frame"],
            "runtime_ms": runtime_ms,
        }
        row.update(metric_row(pred, sample))
        rows.append(row)
        print(f"{method} {sample['frame']}: depth_mae={row['valid_disp_depth_mae_mm']:.3f} runtime={runtime_ms:.1f}ms", flush=True)

    write_csv(args.out_dir / "metrics.csv", rows)
    summary = summarize(rows)
    summary.update({"method": method, "checkpoint": checkpoint, "input_resolution": input_resolution, "frames": len(rows)})
    if device.type == "cuda":
        summary["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated() / 1024**2)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
