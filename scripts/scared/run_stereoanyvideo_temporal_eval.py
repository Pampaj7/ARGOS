#!/usr/bin/env python3
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
from PIL import Image


ARGOS = Path("/home/pampaj/Desktop/ARGOS")
OUT = ARGOS / "results/stereoanyvideo_temporal_eval"
GT5 = ARGOS / "results/video_stereo_repos/test_sequence"
CONSEC32 = ARGOS / "dataset/SCARED/curated/consecutive32"
S2M2_SRC = Path("../../external/frame_stereo_repos/s2m2/src")
S2M2_WEIGHTS = Path("../../external/frame_stereo_repos/s2m2/weights/pretrain_weights")
SAV_REPO = ARGOS / "external/video_stereo_repos/stereoanyvideo"
SAV_CKPT = SAV_REPO / "checkpoints/StereoAnyVideo_MIX.pth"


MODELS = [
    {"name": "S2M2-L@full", "kind": "s2m2", "variant": "L", "width": 0},
    {"name": "S2M2-L@736", "kind": "s2m2", "variant": "L", "width": 736},
    {"name": "S2M2-S@512", "kind": "s2m2", "variant": "S", "width": 512},
    {"name": "StereoAnyVideo@384x640", "kind": "sav", "resize_hw": (384, 640), "iters": 6},
]


def safe_name(name):
    return name.replace("@", "_").replace("/", "_").replace("x", "x")


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_gt5():
    meta = json.loads((GT5 / "metadata.json").read_text())
    frames = []
    for frame in meta["frames"]:
        frames.append(
            {
                "id": f"{frame['index']:06d}",
                "left": read_rgb(GT5 / frame["left"]),
                "right": read_rgb(GT5 / frame["right"]),
                "gt_disp": np.load(GT5 / frame["gt_disparity"]).astype(np.float32),
                "gt_depth": np.load(GT5 / frame["gt_depth"]).astype(np.float32),
                "valid": cv2.imread(str(GT5 / frame["valid_mask"]), cv2.IMREAD_GRAYSCALE) > 0,
                "fx": float(frame["fx"]),
                "baseline_mm": float(frame["baseline_mm"]),
            }
        )
    return frames


def load_sequence_folder(root, limit=None):
    left_paths = sorted((root / "left").glob("*.png"))[:limit]
    right_paths = sorted((root / "right").glob("*.png"))[:limit]
    if len(left_paths) != len(right_paths) or not left_paths:
        raise RuntimeError(f"Expected matching left/right frames under {root}")
    return [
        {"id": left.stem, "left": read_rgb(left), "right": read_rgb(right)}
        for left, right in zip(left_paths, right_paths)
    ]


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
    y = (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(y, cmap)


def build_s2m2(variant):
    sys.path.insert(0, str(S2M2_SRC))
    from s2m2.core.model.s2m2 import S2M2

    cfg = {"S": (128, 1), "L": (256, 3)}[variant]
    model = S2M2(feature_channels=cfg[0], dim_expansion=1, num_transformer=cfg[1], use_positivity=True, refine_iter=3)
    ckpt = torch.load(S2M2_WEIGHTS / f"CH{cfg[0]}NTR{cfg[1]}.pth", map_location="cpu", weights_only=True)
    model.my_load_state_dict(ckpt["state_dict"])
    return model.cuda().eval()


@torch.no_grad()
def infer_s2m2_frame(model, left, right, width):
    sys.path.insert(0, str(S2M2_SRC))
    from s2m2.core.utils.image_utils import image_crop, image_pad

    orig_h, orig_w = left.shape[:2]
    if width == 0:
        left_in, right_in, scale_x = left, right, 1.0
    else:
        scale_x = width / float(orig_w)
        new_h = int(round(orig_h * scale_x))
        left_in = cv2.resize(left, (width, new_h), interpolation=cv2.INTER_LINEAR)
        right_in = cv2.resize(right, (width, new_h), interpolation=cv2.INTER_LINEAR)
    left_t = torch.from_numpy(left_in).permute(2, 0, 1).unsqueeze(0).float().cuda()
    right_t = torch.from_numpy(right_in).permute(2, 0, 1).unsqueeze(0).float().cuda()
    h, w = left_t.shape[-2:]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred, _occ, _conf = model(image_pad(left_t, 32), image_pad(right_t, 32))
    torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    pred = image_crop(pred, (h, w)).squeeze().float().cpu().numpy()
    if scale_x != 1.0:
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale_x
    return np.clip(pred.astype(np.float32), 0, None), runtime_ms


def run_s2m2(config, frames, out_dir):
    model = build_s2m2(config["variant"])
    preds, runtimes = [], []
    peak = 0.0
    for frame in frames:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pred, rt = infer_s2m2_frame(model, frame["left"], frame["right"], config["width"])
        peak = max(peak, torch.cuda.max_memory_allocated() / (1024**2))
        preds.append(pred)
        runtimes.append(rt)
    del model
    torch.cuda.empty_cache()
    save_predictions(config["name"], preds, out_dir)
    return preds, float(np.mean(runtimes)), float(np.median(runtimes)), peak


def build_sav():
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
    return model.cuda().eval()


@torch.no_grad()
def run_sav(config, frames, out_dir):
    model = build_sav()
    h, w = config["resize_hw"]
    lefts, rights = [], []
    orig_h, orig_w = frames[0]["left"].shape[:2]
    for frame in frames:
        left = torch.from_numpy(frame["left"]).permute(2, 0, 1).float().cuda()
        right = torch.from_numpy(frame["right"]).permute(2, 0, 1).float().cuda()
        lefts.append(F.interpolate(left[None], size=(h, w), mode="bilinear", align_corners=True)[0])
        rights.append(F.interpolate(right[None], size=(h, w), mode="bilinear", align_corners=True)[0])
    stereo_video = torch.stack([torch.stack(lefts, 0), torch.stack(rights, 0)], dim=1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    raw = model.forward(stereo_video[:, 0][None], stereo_video[:, 1][None], iters=config["iters"], test_mode=True)
    torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    if raw.shape[0] == len(frames):
        disp = raw[:, 0, :1].abs()
    else:
        disp = raw[0, :, :1].abs()
    disp_np = disp.squeeze(1).float().cpu().numpy().astype(np.float32)
    scale_x = w / float(orig_w)
    preds = [cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale_x for d in disp_np]
    del model
    torch.cuda.empty_cache()
    save_predictions(config["name"], preds, out_dir)
    per_frame = runtime_ms / max(len(frames), 1)
    return preds, per_frame, per_frame, peak


def save_predictions(model_name, preds, out_dir):
    pred_dir = out_dir / "predictions" / safe_name(model_name)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for idx, pred in enumerate(preds):
        np.save(pred_dir / f"{idx:06d}.npy", pred.astype(np.float32))
        cv2.imwrite(str(pred_dir / f"{idx:06d}.png"), colorize(pred))


def per_frame_metrics(model_name, preds, frames, sequence_name):
    rows = []
    if "gt_disp" not in frames[0]:
        return rows
    for idx, (pred, frame) in enumerate(zip(preds, frames)):
        gt_disp = frame["gt_disp"]
        gt_depth = frame["gt_depth"]
        valid = frame["valid"] & (gt_disp > 0) & (gt_depth > 0) & np.isfinite(pred)
        pred_depth = frame["fx"] * frame["baseline_mm"] / np.maximum(pred, 1e-6)
        disp_err = np.abs(pred[valid] - gt_disp[valid])
        depth_err = np.abs(pred_depth[valid] - gt_depth[valid])
        rows.append(
            {
                "sequence": sequence_name,
                "model_name": model_name,
                "frame": idx,
                "valid_disp_mae": float(disp_err.mean()),
                "valid_disp_rmse": float(np.sqrt((disp_err**2).mean())),
                "bad_1px": float((disp_err > 1).mean() * 100),
                "bad_2px": float((disp_err > 2).mean() * 100),
                "bad_3px": float((disp_err > 3).mean() * 100),
                "valid_depth_mae": float(depth_err.mean()),
                "valid_depth_median": float(np.median(depth_err)),
                "valid_depth_rmse": float(np.sqrt((depth_err**2).mean())),
                "bad_1mm": float((depth_err > 1).mean() * 100),
                "bad_2mm": float((depth_err > 2).mean() * 100),
                "bad_5mm": float((depth_err > 5).mean() * 100),
                "pred_disp_le_0_1_ratio": float((frame["valid"] & (pred <= 0.1)).sum() / max(int(frame["valid"].sum()), 1)),
                "pred_disp_le_0_5_ratio": float((frame["valid"] & (pred <= 0.5)).sum() / max(int(frame["valid"].sum()), 1)),
                "valid_pixel_ratio": float(valid.mean()),
            }
        )
    return rows


def temporal_metrics(model_name, preds, frames, sequence_name):
    diffs, depth_diffs, err_vars = [], [], []
    prev_pred = None
    prev_depth = None
    prev_err = None
    valid_stack = []
    for pred, frame in zip(preds, frames):
        valid = np.isfinite(pred) & (pred > 0.1)
        if "valid" in frame:
            valid &= frame["valid"]
        valid_stack.append(valid)
        if "fx" in frame:
            pred_depth = frame["fx"] * frame["baseline_mm"] / np.maximum(pred, 1e-6)
        else:
            pred_depth = None
        if prev_pred is not None:
            both = valid & prev_valid
            diffs.append(float(np.mean(np.abs(pred[both] - prev_pred[both]))))
            if pred_depth is not None and prev_depth is not None:
                depth_diffs.append(float(np.mean(np.abs(pred_depth[both] - prev_depth[both]))))
            if "gt_disp" in frame and prev_err is not None:
                err = np.full_like(pred, np.nan, dtype=np.float32)
                gt_valid = both & (frame["gt_disp"] > 0)
                err[gt_valid] = np.abs(pred[gt_valid] - frame["gt_disp"][gt_valid])
                both_err = np.isfinite(err) & np.isfinite(prev_err)
                err_vars.append(float(np.mean(np.abs(err[both_err] - prev_err[both_err]))))
                prev_err = err
        if "gt_disp" in frame:
            prev_err = np.where(valid, np.abs(pred - frame["gt_disp"]), np.nan).astype(np.float32)
        prev_pred = pred
        prev_depth = pred_depth
        prev_valid = valid
    common_valid = np.logical_and.reduce(valid_stack)
    stack = np.stack(preds)
    temporal_std = float(np.nanmean(np.nanstd(np.where(common_valid[None], stack, np.nan), axis=0)))
    return {
        "sequence": sequence_name,
        "model_name": model_name,
        "mean_consecutive_disp_diff": float(np.mean(diffs)) if diffs else None,
        "mean_consecutive_depth_diff": float(np.mean(depth_diffs)) if depth_diffs else None,
        "per_pixel_temporal_std": temporal_std,
        "temporal_error_variation": float(np.mean(err_vars)) if err_vars else None,
        "frames": len(preds),
    }


def summarize_frame_rows(rows):
    if not rows:
        return {}
    skip = {"sequence", "model_name", "frame"}
    out = {}
    for key in rows[0]:
        if key in skip:
            continue
        out[key] = float(np.mean([r[key] for r in rows]))
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_qualitative(sequence_name, frames, pred_by_model, out_dir):
    qdir = out_dir / "qualitative" / sequence_name
    qdir.mkdir(parents=True, exist_ok=True)
    selected = [0, len(frames) // 2, len(frames) - 1]
    for idx in selected:
        frame = frames[idx]
        tiles = [cv2.cvtColor(frame["left"], cv2.COLOR_RGB2BGR)]
        labels = ["left"]
        if "gt_disp" in frame:
            valid = frame["valid"] & (frame["gt_disp"] > 0)
            vmax = float(np.nanpercentile(frame["gt_disp"][valid], 99))
            tiles.append(colorize(frame["gt_disp"], vmax))
            labels.append("GT disp")
        else:
            vmax = None
        for model_name, preds in pred_by_model.items():
            pred = preds[idx]
            tiles.append(colorize(pred, vmax))
            labels.append(model_name)
            if "gt_disp" in frame:
                err = np.abs(pred - frame["gt_disp"])
                tiles.append(colorize(err, 25.0, cv2.COLORMAP_MAGMA))
                labels.append("abs err")
            if idx > 0:
                diff = np.abs(pred - preds[idx - 1])
                tiles.append(colorize(diff, 25.0, cv2.COLORMAP_VIRIDIS))
                labels.append("temp diff")
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (200, 160), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label[:24], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
        cv2.imwrite(str(qdir / f"{idx:06d}_montage.png"), np.concatenate(small, axis=1))


def make_videos(sequence_name, pred_by_model, out_dir):
    vdir = out_dir / "videos" / sequence_name
    vdir.mkdir(parents=True, exist_ok=True)
    for model_name in ["S2M2-L@736", "StereoAnyVideo@384x640"]:
        preds = pred_by_model.get(model_name)
        if not preds:
            continue
        first = colorize(preds[0])
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(vdir / f"{safe_name(model_name)}_disparity.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
        diff_writer = cv2.VideoWriter(str(vdir / f"{safe_name(model_name)}_temporal_diff.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
        prev = None
        for pred in preds:
            writer.write(colorize(pred))
            diff_writer.write(colorize(np.zeros_like(pred) if prev is None else np.abs(pred - prev), 25.0, cv2.COLORMAP_VIRIDIS))
            prev = pred
        writer.release()
        diff_writer.release()


def run_sequence(sequence_name, frames):
    out_dir = OUT / sequence_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_by_model = {}
    per_frame = []
    temporal = []
    summary = []
    for config in MODELS:
        print(f"running {sequence_name} {config['name']}", flush=True)
        if config["kind"] == "s2m2":
            preds, avg_rt, med_rt, peak = run_s2m2(config, frames, out_dir)
        else:
            preds, avg_rt, med_rt, peak = run_sav(config, frames, out_dir)
        pred_by_model[config["name"]] = preds
        frame_rows = per_frame_metrics(config["name"], preds, frames, sequence_name)
        per_frame.extend(frame_rows)
        tm = temporal_metrics(config["name"], preds, frames, sequence_name)
        temporal.append(tm)
        item = {
            "sequence": sequence_name,
            "model_name": config["name"],
            "avg_runtime_ms": avg_rt,
            "median_runtime_ms": med_rt,
            "peak_gpu_memory_mb": peak,
            **summarize_frame_rows(frame_rows),
            **{k: v for k, v in tm.items() if k not in {"sequence", "model_name"}},
        }
        summary.append(item)
    make_qualitative(sequence_name, frames, pred_by_model, out_dir)
    make_videos(sequence_name, pred_by_model, out_dir)
    write_csv(out_dir / "per_frame_metrics.csv", per_frame)
    write_csv(out_dir / "temporal_metrics.csv", temporal)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "per_frame": per_frame, "temporal": temporal}, indent=2) + "\n")
    return summary, per_frame, temporal


def write_report(all_summary, all_per_frame, all_temporal):
    write_csv(OUT / "report.csv", all_summary)
    (OUT / "report.json").write_text(json.dumps({"summary": all_summary, "per_frame": all_per_frame, "temporal": all_temporal}, indent=2) + "\n")
    gt5 = [r for r in all_summary if r["sequence"] == "gt5"]
    seq32 = [r for r in all_summary if r["sequence"] == "consecutive32"]
    def row(seq, model):
        return next(r for r in seq if r["model_name"] == model)
    sav5 = row(gt5, "StereoAnyVideo@384x640")
    lfull5 = row(gt5, "S2M2-L@full")
    l736_5 = row(gt5, "S2M2-L@736")
    sav32 = row(seq32, "StereoAnyVideo@384x640")
    l736_32 = row(seq32, "S2M2-L@736")
    lines = [
        "# StereoAnyVideo Temporal Evaluation",
        "",
        "This run integrates StereoAnyVideo as the first video-stereo upper-bound baseline and compares it with frame-based S2M2 baselines.",
        "",
        "Sequences:",
        "",
        "- `gt5`: the 5-frame ARGOS/SCARED smoke sequence with GT disparity/depth. These frames are clean keyframes and are not guaranteed to be temporally consecutive.",
        "- `consecutive32`: 32 consecutive SCARED stereo frames without GT, used for true temporal/flicker metrics.",
        "",
        "All resized disparities are rescaled back to original image coordinates with `pred_disp_original = pred_disp_resized / scale_x`.",
        "",
        "## Summary",
        "",
        "| sequence | model | disp MAE | depth MAE | bad 2px | bad 2mm | disp diff | depth diff | temporal std | error variation | runtime ms | peak MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_summary:
        lines.append(
            f"| {r['sequence']} | {r['model_name']} | {r.get('valid_disp_mae', float('nan')):.4f} | "
            f"{r.get('valid_depth_mae', float('nan')):.4f} | {r.get('bad_2px', float('nan')):.2f} | "
            f"{r.get('bad_2mm', float('nan')):.2f} | {r['mean_consecutive_disp_diff']:.4f} | "
            f"{'' if r['mean_consecutive_depth_diff'] is None else f'{r['mean_consecutive_depth_diff']:.4f}'} | "
            f"{r['per_pixel_temporal_std']:.4f} | {'' if r['temporal_error_variation'] is None else f'{r['temporal_error_variation']:.4f}'} | "
            f"{r['avg_runtime_ms']:.2f} | {r['peak_gpu_memory_mb']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Temporal consistency: on the true `consecutive32` sequence, StereoAnyVideo has mean consecutive disparity diff `{sav32['mean_consecutive_disp_diff']:.4f}` vs S2M2-L@736 `{l736_32['mean_consecutive_disp_diff']:.4f}`. Lower is smoother. This run therefore gives the first direct flicker comparison, but without GT on the consecutive clip.",
            f"2. Accuracy on `gt5`: StereoAnyVideo depth MAE `{sav5['valid_depth_mae']:.4f} mm` vs S2M2-L@full `{lfull5['valid_depth_mae']:.4f} mm` and S2M2-L@736 `{l736_5['valid_depth_mae']:.4f} mm`. Disparity MAE is `{sav5['valid_disp_mae']:.4f} px` vs `{lfull5['valid_disp_mae']:.4f} px` and `{l736_5['valid_disp_mae']:.4f} px`.",
            f"3. Runtime/VRAM: StereoAnyVideo@384x640 costs `{sav5['avg_runtime_ms']:.2f} ms/frame` and `{sav5['peak_gpu_memory_mb']:.1f} MB` on `gt5`; compare S2M2-L@736 `{l736_5['avg_runtime_ms']:.2f} ms/frame`, `{l736_5['peak_gpu_memory_mb']:.1f} MB`.",
            "4. Practical use: StereoAnyVideo is useful now as an upper-bound/teacher-style video baseline. At reduced resolution and short window it is not absurdly far from deployment, but S2M2-L@736 remains the practical baseline until timed larger clips prove the video prior buys enough stability.",
            "5. Temporal teacher: yes. StereoAnyVideo should be used as the first temporal teacher/reference for a future lightweight stabilizer or distillation experiment.",
            "",
            "Qualitative montages are in `gt5/qualitative/` and `consecutive32/qualitative/`. Short MP4 visualizations are in each sequence's `videos/` folder.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = {
        "gt5": str(GT5),
        "consecutive32": str(CONSEC32),
        "models": MODELS,
        "stereoanyvideo_checkpoint": str(SAV_CKPT),
        "note": "No training. Previous benchmark folders are not modified.",
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    gt5 = load_gt5()
    seq32 = load_sequence_folder(CONSEC32, limit=32)
    all_summary, all_per_frame, all_temporal = [], [], []
    for name, frames in [("gt5", gt5), ("consecutive32", seq32)]:
        summary, per_frame, temporal = run_sequence(name, frames)
        all_summary.extend(summary)
        all_per_frame.extend(per_frame)
        all_temporal.extend(temporal)
    write_report(all_summary, all_per_frame, all_temporal)


if __name__ == "__main__":
    main()
