#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.temporal_refinement.evaluate_scared_temporal_gt as gt_eval
from scripts.temporal_refinement.extend_temporal_gt_fairness_checks import (
    calibration_fx_baseline,
    compute_flow,
    compute_geom,
    depth_from_disp,
    ema_sequence,
    load_rgb,
    mean,
    warp_prev_to_current,
    write_csv,
)

ALPHA = 0.50
MIN_VALID_RATIO = 0.20


@dataclass(frozen=True)
class Frame:
    sequence_id: str
    frame_id: str
    frame_index: int
    left_path: Path
    right_path: Path
    gt_disp_path: Path
    gt_depth_path: Path
    valid_mask_path: Path
    calib_path: Path


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def frame_from_row(row: dict[str, str]) -> Frame:
    if row.get('sequence_id'):
        sequence_id = row['sequence_id']
        frame_id = row['frame_id']
    else:
        sequence_id = f"{row['dataset_id']}_{row['keyframe_id']}"
        frame_id = f"{int(row['frame_id']):06d}"
    return Frame(
        sequence_id=sequence_id,
        frame_id=frame_id,
        frame_index=int(row['frame_id']),
        left_path=resolve_path(row['left_path']),
        right_path=resolve_path(row['right_path']),
        gt_disp_path=resolve_path(row['disparity_float32_path']),
        gt_depth_path=resolve_path(row['depth_float32_path']),
        valid_mask_path=resolve_path(row['valid_mask_path']),
        calib_path=resolve_path(row['calibration_path']),
    )


def load_candidates() -> dict[str, list[Frame]]:
    candidates: dict[str, list[Frame]] = defaultdict(list)
    for csv_path in [
        ROOT / 'dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3/metadata.csv',
        ROOT / 'dataset/SCARED/curated/warped_gt_108/metadata.csv',
    ]:
        for row in read_rows(csv_path):
            valid_ratio = float(row.get('valid_pixel_ratio', '0') or 0)
            if valid_ratio < MIN_VALID_RATIO:
                continue
            frame = frame_from_row(row)
            candidates[frame.sequence_id].append(frame)
    return {k: sorted(v, key=lambda f: f.frame_index) for k, v in candidates.items()}


def audit_sequence(sequence_id: str, frames: list[Frame]) -> dict[str, object]:
    indices = [f.frame_index for f in frames]
    duplicate_count = len(indices) - len(set(indices))
    missing = []
    if indices:
        expected = list(range(min(indices), max(indices) + 1))
        missing = sorted(set(expected) - set(indices))
    files_ok = all(
        p.exists()
        for f in frames
        for p in [f.left_path, f.right_path, f.gt_disp_path, f.gt_depth_path, f.valid_mask_path, f.calib_path]
    )
    calibs_ok = True
    shapes_ok = True
    diffs = []
    prev = None
    for f in frames:
        try:
            calib = json.loads(f.calib_path.read_text())
            _fx, _baseline = calibration_fx_baseline(f.calib_path)
            rgb = load_rgb(f.left_path)
            gt = np.load(f.gt_disp_path)
            mask = np.load(f.valid_mask_path)
            shapes_ok = shapes_ok and rgb.shape[:2] == gt.shape == mask.shape
            if prev is not None:
                diffs.append(float(np.mean(np.abs(rgb.astype(np.float32) - prev.astype(np.float32)))))
            prev = rgb
            calibs_ok = calibs_ok and bool(calib)
        except Exception:
            calibs_ok = False
            shapes_ok = False
    median_adjacent_rgb_absdiff = float(np.median(diffs)) if diffs else math.nan
    valid_for_temporal = (
        len(frames) >= 5
        and duplicate_count == 0
        and not missing
        and files_ok
        and calibs_ok
        and shapes_ok
        and math.isfinite(median_adjacent_rgb_absdiff)
        and 0.1 < median_adjacent_rgb_absdiff < 80.0
    )
    reason = 'ok' if valid_for_temporal else []
    if not valid_for_temporal:
        reasons = []
        if len(frames) < 5:
            reasons.append('too_few_frames')
        if duplicate_count:
            reasons.append('duplicate_frame_indices')
        if missing:
            reasons.append(f'missing_frames:{missing[:10]}')
        if not files_ok:
            reasons.append('missing_files')
        if not calibs_ok:
            reasons.append('bad_calibration')
        if not shapes_ok:
            reasons.append('shape_mismatch')
        if not math.isfinite(median_adjacent_rgb_absdiff) or not (0.1 < median_adjacent_rgb_absdiff < 80.0):
            reasons.append('failed_temporal_continuity_rgb_check')
        reason = ';'.join(reasons)
    return {
        'sequence_id': sequence_id,
        'num_frames': len(frames),
        'first_frame': min(indices) if indices else '',
        'last_frame': max(indices) if indices else '',
        'duplicates': duplicate_count,
        'missing_frames': len(missing),
        'files_ok': files_ok,
        'calibration_and_gt_ok': calibs_ok and shapes_ok,
        'median_adjacent_rgb_absdiff': median_adjacent_rgb_absdiff,
        'valid_for_temporal_eval': valid_for_temporal,
        'reason': reason,
    }


def prediction_name(sequence_id: str, frame_id: str) -> str:
    return f'{sequence_id}_frame_{int(frame_id):06d}_disp.npy'


def load_existing_prediction(method_dir: Path, sequence_id: str, frame_id: str) -> np.ndarray:
    candidates = [
        method_dir / f'{frame_id}.npy',
        method_dir / prediction_name(sequence_id, frame_id),
        method_dir / f'test_dataset_9_keyframe_3_frame_{int(frame_id):06d}_disp.npy',
    ]
    for p in candidates:
        if p.exists():
            return np.load(p).astype(np.float32)
    matches = list(method_dir.glob(f'*{sequence_id}*frame_{int(frame_id):06d}_disp.npy')) + list(method_dir.glob(f'*frame_{int(frame_id):06d}_disp.npy'))
    if matches:
        return np.load(matches[0]).astype(np.float32)
    raise FileNotFoundError(f'{method_dir} {sequence_id} {frame_id}')


def to_eval_frame(frame: Frame) -> dict:
    fx, baseline = calibration_fx_baseline(frame.calib_path)
    return {
        'id': f'{frame.sequence_id}_frame_{frame.frame_id}',
        'left_path': frame.left_path,
        'right_path': frame.right_path,
        'gt_disp_path': frame.gt_disp_path,
        'gt_depth_path': frame.gt_depth_path,
        'valid_mask_path': frame.valid_mask_path,
        'calib_path': frame.calib_path,
        'fx': fx,
        'baseline_mm': baseline,
    }


def configure_eval_module() -> None:
    stereo_root = ROOT.parent / 'stereo'
    gt_eval.S2M2_REPO = stereo_root / 's2m2'
    gt_eval.S2M2_SRC = gt_eval.S2M2_REPO / 'src'
    gt_eval.S2M2_WEIGHTS = gt_eval.S2M2_REPO / 'weights/pretrain_weights'
    gt_eval.SAV_REPO = stereo_root / 'stereoanyvideo'
    gt_eval.SAV_CKPT = gt_eval.SAV_REPO / 'checkpoints/StereoAnyVideo_MIX.pth'


def run_stereoanyvideo_sequence(frames: list[Frame], device: torch.device) -> tuple[list[np.ndarray], float, float]:
    eval_frames = [to_eval_frame(f) for f in frames]
    model = gt_eval.build_sav(device)
    output: list[np.ndarray | None] = [None] * len(eval_frames)
    runtimes: list[float] = []
    peak = 0.0
    chunk_size = 32
    overlap = 4
    cursor = 0
    written: set[int] = set()
    while cursor < len(eval_frames):
        end = min(cursor + chunk_size, len(eval_frames))
        chunk = eval_frames[cursor:end]
        chunk_preds, runtime, chunk_peak = gt_eval.infer_sav_chunk(model, chunk, (384, 640), 6, device)
        runtimes.extend([runtime] * len(chunk))
        peak = max(peak, chunk_peak)
        for local_idx, pred in enumerate(chunk_preds):
            global_idx = cursor + local_idx
            if global_idx not in written:
                output[global_idx] = np.clip(pred.astype(np.float32), 0, None)
                written.add(global_idx)
        if end >= len(eval_frames):
            break
        cursor += max(1, chunk_size - overlap)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if any(p is None for p in output):
        raise RuntimeError('StereoAnyVideo chunking missed frames')
    return [p for p in output if p is not None], float(np.mean(runtimes)), peak


def run_convgru_sequence(frames: list[Frame], raw_l: list[np.ndarray], device: torch.device) -> tuple[list[np.ndarray], float, float]:
    ckpt = ROOT / 'results/03_temporal_refinement/training/convgru/temporal_refinement_train_convgru_l736_v2_scheduled/checkpoints/epoch_0040.pt'
    model, _ = gt_eval.load_checkpoint(ckpt, 'convgru', device)
    hidden = None
    preds = []
    runtimes = []
    peak = 0.0
    for frame, raw in zip(frames, raw_l):
        rgb = load_rgb(frame.left_path).astype(np.float32) / 255.0
        x = np.concatenate([rgb.transpose(2, 0, 1), raw[None] / 128.0], axis=0)
        x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)
        center = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).float().to(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                delta, hidden = model(x_t, hidden)
                refined = torch.clamp(center + delta, min=0.0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            peak = max(peak, torch.cuda.max_memory_allocated() / (1024 ** 2))
        runtimes.append((time.perf_counter() - t0) * 1000.0)
        preds.append(refined[0, 0].detach().float().cpu().numpy().astype(np.float32))
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return preds, float(np.mean(runtimes)), peak


def common_masks(preds: dict[str, list[np.ndarray]], masks: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for i, mask in enumerate(masks):
        cm = mask.copy()
        for seq in preds.values():
            p = seq[i]
            cm &= np.isfinite(p) & (p > 0.1)
        out.append(cm)
    return out


def temporal_metrics(seq: list[np.ndarray], frames: list[Frame], masks: list[np.ndarray], flows: list[np.ndarray | None]) -> tuple[float, float]:
    raw_vals = []
    mc_vals = []
    for i in range(1, len(seq)):
        prev, cur = seq[i - 1], seq[i]
        raw_mask = masks[i - 1] & masks[i] & np.isfinite(prev) & np.isfinite(cur) & (prev > 0.1) & (cur > 0.1)
        if raw_mask.any():
            raw_vals.append(float(np.mean(np.abs(cur[raw_mask] - prev[raw_mask]))))
        flow = flows[i]
        if flow is None:
            continue
        warped = warp_prev_to_current(prev, flow)
        mc_mask = masks[i] & np.isfinite(warped) & np.isfinite(cur) & (warped > 0.1) & (cur > 0.1)
        if mc_mask.any():
            mc_vals.append(float(np.mean(np.abs(cur[mc_mask] - warped[mc_mask]))))
    return mean(raw_vals), mean(mc_vals)


def evaluate_method(sequence_id: str, method: str, seq: list[np.ndarray], frames: list[Frame], cmasks: list[np.ndarray], flows: list[np.ndarray | None], runtime: float, vram: float) -> dict[str, object]:
    geom = []
    for i, frame in enumerate(frames):
        gt_disp = np.load(frame.gt_disp_path).astype(np.float32)
        gt_depth = np.load(frame.gt_depth_path).astype(np.float32)
        fx, baseline = calibration_fx_baseline(frame.calib_path)
        geom.append(compute_geom(seq[i], gt_disp, gt_depth, cmasks[i], fx, baseline))
    raw_t, mc_t = temporal_metrics(seq, frames, cmasks, flows)
    return {
        'sequence_id': sequence_id,
        'method': method,
        'frames': len(frames),
        'depth_mae_mm': mean([g['depth_mae'] for g in geom]),
        'disp_mae_px': mean([g['disp_mae'] for g in geom]),
        'bad_2mm_pct': mean([g['bad_2mm'] for g in geom]),
        'raw_temporal_diff': raw_t,
        'motion_compensated_temporal_mae': mc_t,
        'runtime_ms': runtime,
        'peak_vram_mb': vram,
    }


def weighted_overall(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        groups[str(r['method'])].append(r)
    overall = []
    for method, rs in groups.items():
        weights = np.array([float(r['frames']) for r in rs], dtype=np.float64)
        row = {'method': method, 'sequences': len(rs), 'frames': int(weights.sum())}
        for key in ['depth_mae_mm','disp_mae_px','bad_2mm_pct','raw_temporal_diff','motion_compensated_temporal_mae','runtime_ms','peak_vram_mb']:
            vals = np.array([float(r[key]) for r in rs], dtype=np.float64)
            finite = np.isfinite(vals)
            row[key] = float(np.average(vals[finite], weights=weights[finite])) if finite.any() else math.nan
        overall.append(row)
    return sorted(overall, key=lambda r: (float(r['depth_mae_mm']), float(r['motion_compensated_temporal_mae'])))


def save_contact_sheet(sequence_id: str, frames: list[Frame], out_dir: Path) -> Path:
    imgs = [load_rgb(f.left_path) for f in frames[:6]]
    tiles = []
    for i, img in enumerate(imgs):
        tile = cv2.resize(img, (180, 144), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (180, 22), (0, 0, 0), -1)
        cv2.putText(tile, frames[i].frame_id, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.concatenate(tiles, axis=1)
    path = out_dir / f'{sequence_id}_adjacent_contact.png'
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return path


def colorize(disp: np.ndarray, vmax: float) -> np.ndarray:
    x = np.clip(disp / max(vmax, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(x, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def save_comparison(sequence_id: str, frames: list[Frame], preds: dict[str, list[np.ndarray]], out_dir: Path, suffix: str) -> Path:
    idx = len(frames) // 2
    rgb = load_rgb(frames[idx].left_path)
    methods = ['S2M2-S@512','S2M2-S@512+EMA0.50','ConvGRU V2 e40','StereoAnyVideo']
    vmax = float(np.nanpercentile(np.concatenate([preds[m][idx].ravel() for m in methods]), 98))
    tiles = [cv2.resize(rgb, (220,176), interpolation=cv2.INTER_AREA)]
    labels = ['RGB']
    for m in methods:
        tiles.append(cv2.resize(colorize(preds[m][idx], vmax), (220,176), interpolation=cv2.INTER_AREA))
        labels.append(m)
    for tile, label in zip(tiles, labels):
        cv2.rectangle(tile, (0,0), (220,24), (0,0,0), -1)
        cv2.putText(tile, label[:24], (5,17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255,255,255), 1, cv2.LINE_AA)
    canvas = np.concatenate(tiles, axis=1)
    path = out_dir / f'{suffix}_{sequence_id}_comparison.png'
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return path


def save_motion_error(sequence_id: str, frames: list[Frame], preds: dict[str, list[np.ndarray]], out_dir: Path) -> Path:
    if len(frames) < 2:
        raise ValueError('too few frames')
    i = len(frames) // 2
    rgb_prev = load_rgb(frames[i-1].left_path)
    rgb_cur = load_rgb(frames[i].left_path)
    flow = compute_flow(rgb_prev, rgb_cur)
    method = 'S2M2-S@512+EMA0.50'
    warped = warp_prev_to_current(preds[method][i-1], flow)
    err = np.abs(preds[method][i] - warped)
    tile = colorize(np.nan_to_num(err, nan=0.0), float(np.nanpercentile(err, 98)))
    path = out_dir / f'{sequence_id}_motion_comp_error.png'
    cv2.imwrite(str(path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'results/03_temporal_refinement/evaluation/scared_multisequence_compact')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = args.out_dir
    ref_dir = out / 'reference_images'
    ref_dir.mkdir(parents=True, exist_ok=True)
    configure_eval_module()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    candidates = load_candidates()
    audit_rows = [audit_sequence(sid, frames) for sid, frames in candidates.items()]
    included = {r['sequence_id']: candidates[str(r['sequence_id'])] for r in audit_rows if r['valid_for_temporal_eval']}
    write_csv(out / 'sequence_audit.csv', audit_rows, ['sequence_id','num_frames','first_frame','last_frame','duplicates','missing_frames','files_ok','calibration_and_gt_ok','median_adjacent_rgb_absdiff','valid_for_temporal_eval','reason'])

    per_sequence = []
    reference_paths = []
    runtimes = {
        'S2M2-S@512': 60.974148300906215,
        'S2M2-L@736': 187.76713038364855,
        'Fast-FoundationStereo ONNX': 45.3597867449459,
    }
    vrams = {
        'S2M2-S@512': 371.33447265625,
        'S2M2-L@736': 1672.4599609375,
        'Fast-FoundationStereo ONNX': math.nan,
    }

    temporal_dirs = {
        'S2M2-S@512': ROOT / 'results/03_temporal_refinement/evaluation/gt_temporal_test_dataset_9_keyframe_3/predictions/S2M2-S_512',
        'S2M2-L@736': ROOT / 'results/03_temporal_refinement/evaluation/gt_temporal_test_dataset_9_keyframe_3/predictions/S2M2-L_736',
        'Fast-FoundationStereo ONNX': ROOT / 'results/03_temporal_refinement/evaluation/frame_based_gt/native_frame_methods/Fast-FoundationStereo_ONNX',
    }
    warped_dirs = {
        'S2M2-S@512': ROOT / 'results/01_frame_stereo/SCARED/warped_gt_108/S2M2-S',
        'S2M2-L@736': ROOT / 'results/01_frame_stereo/SCARED/warped_gt_108/S2M2-L',
        'Fast-FoundationStereo ONNX': ROOT / 'results/01_frame_stereo/SCARED/warped_gt_108/Fast-FoundationStereo_ONNX',
    }

    seq_cache: dict[str, dict[str, list[np.ndarray]]] = {}
    for sid, frames in included.items():
        print(f'[{sid}] frames={len(frames)}', flush=True)
        preds = {}
        source_dirs = temporal_dirs if sid == 'test_dataset_9_keyframe_3' else warped_dirs
        for method in ['S2M2-S@512','S2M2-L@736','Fast-FoundationStereo ONNX']:
            preds[method] = [load_existing_prediction(source_dirs[method], sid, f.frame_id) for f in frames]
        preds['S2M2-S@512+EMA0.50'] = ema_sequence(preds['S2M2-S@512'], ALPHA)
        preds['S2M2-L@736+EMA0.50'] = ema_sequence(preds['S2M2-L@736'], ALPHA)
        sav, sav_rt, sav_vram = run_stereoanyvideo_sequence(frames, device)
        preds['StereoAnyVideo'] = sav
        conv, conv_rt, conv_vram = run_convgru_sequence(frames, preds['S2M2-L@736'], device)
        preds['ConvGRU V2 e40'] = conv
        seq_cache[sid] = preds
        masks = [np.load(f.valid_mask_path).astype(bool) for f in frames]
        rgbs = [load_rgb(f.left_path) for f in frames]
        flows: list[np.ndarray | None] = [None] + [compute_flow(rgbs[i - 1], rgbs[i]) for i in range(1, len(rgbs))]
        cmasks = common_masks(preds, masks)
        method_runtime = dict(runtimes)
        method_vram = dict(vrams)
        method_runtime['S2M2-S@512+EMA0.50'] = runtimes['S2M2-S@512']
        method_vram['S2M2-S@512+EMA0.50'] = vrams['S2M2-S@512']
        method_runtime['S2M2-L@736+EMA0.50'] = runtimes['S2M2-L@736']
        method_vram['S2M2-L@736+EMA0.50'] = vrams['S2M2-L@736']
        method_runtime['StereoAnyVideo'] = sav_rt
        method_vram['StereoAnyVideo'] = sav_vram
        method_runtime['ConvGRU V2 e40'] = runtimes['S2M2-L@736'] + conv_rt
        method_vram['ConvGRU V2 e40'] = vrams['S2M2-L@736'] + conv_vram
        for method, seq in preds.items():
            per_sequence.append(evaluate_method(sid, method, seq, frames, cmasks, flows, method_runtime[method], method_vram[method]))
        reference_paths.append(str(save_contact_sheet(sid, frames, ref_dir)))

    overall = weighted_overall(per_sequence)
    write_csv(out / 'per_sequence_metrics.csv', per_sequence, ['sequence_id','method','frames','depth_mae_mm','disp_mae_px','bad_2mm_pct','raw_temporal_diff','motion_compensated_temporal_mae','runtime_ms','peak_vram_mb'])
    write_csv(out / 'overall_summary.csv', overall, ['method','sequences','frames','depth_mae_mm','disp_mae_px','bad_2mm_pct','raw_temporal_diff','motion_compensated_temporal_mae','runtime_ms','peak_vram_mb'])

    # Reference images: best, median, worst sequence by S2M2-S EMA depth MAE.
    s_rows = sorted([r for r in per_sequence if r['method'] == 'S2M2-S@512+EMA0.50'], key=lambda r: float(r['depth_mae_mm']))
    picks = [('best', s_rows[0]), ('median', s_rows[len(s_rows)//2]), ('worst', s_rows[-1])]
    for label, row in picks:
        sid = str(row['sequence_id'])
        reference_paths.append(str(save_comparison(sid, included[sid], seq_cache[sid], ref_dir, label)))
    reference_paths.append(str(save_motion_error(str(s_rows[len(s_rows)//2]['sequence_id']), included[str(s_rows[len(s_rows)//2]['sequence_id'])], seq_cache[str(s_rows[len(s_rows)//2]['sequence_id'])], ref_dir)))

    deployment = next(r for r in overall if r['method'] == 'S2M2-S@512+EMA0.50')
    conv = next(r for r in overall if r['method'] == 'ConvGRU V2 e40')
    remains_best = float(deployment['depth_mae_mm']) <= float(conv['depth_mae_mm']) and float(deployment['motion_compensated_temporal_mae']) <= float(conv['motion_compensated_temporal_mae'])
    confirmed = [sid for sid, frames in included.items()]
    excluded = [r for r in audit_rows if not r['valid_for_temporal_eval']]
    report_lines = [
        '# Compact SCARED Multi-Sequence Generalization Benchmark',
        '',
        '## Confirmed Progressive Sequences',
        '',
        *[f'- `{sid}` ({len(included[sid])} frames)' for sid in confirmed],
        '',
        '## Excluded Sequences',
        '',
    ]
    if excluded:
        report_lines.extend([f"- `{r['sequence_id']}`: {r['reason']}" for r in excluded])
    else:
        report_lines.append('- None')
    report_lines.extend(['', '## Overall Summary', '', '| method | depth MAE | disp MAE | Bad-2mm | raw temporal | motion-comp temporal | runtime ms | VRAM MB |', '| --- | --- | --- | --- | --- | --- | --- | --- |'])
    for r in overall:
        report_lines.append(f"| {r['method']} | {r['depth_mae_mm']:.4f} | {r['disp_mae_px']:.4f} | {r['bad_2mm_pct']:.4f} | {r['raw_temporal_diff']:.4f} | {r['motion_compensated_temporal_mae']:.4f} | {r['runtime_ms']:.2f} | {r['peak_vram_mb']:.1f} |")
    report_lines.extend([
        '',
        '## Answer',
        '',
        f"S2M2-S@512 + EMA alpha 0.50 remains the best deployment-oriented configuration: `{remains_best}`.",
        f"It uses `{deployment['runtime_ms']:.2f} ms` and `{deployment['peak_vram_mb']:.1f} MB` versus ConvGRU e40 `{conv['runtime_ms']:.2f} ms` and `{conv['peak_vram_mb']:.1f} MB`.",
        '',
        '## Reference Images',
        '',
        *[f'- `{p}`' for p in reference_paths],
    ])
    (out / 'report.md').write_text('\n'.join(report_lines) + '\n')
    (out / 'run.log').write_text(f'device={device}\nincluded={len(included)}\nalpha={ALPHA}\n')
    print(f'Wrote {out}', flush=True)
    print(f'remains_best={remains_best}', flush=True)


if __name__ == '__main__':
    main()
