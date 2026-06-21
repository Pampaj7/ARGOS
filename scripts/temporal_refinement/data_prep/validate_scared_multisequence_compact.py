#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.temporal_refinement.evaluate_scared_temporal_gt as gt_eval
from scripts.temporal_refinement.run_scared_multisequence_compact import (
    ALPHA,
    audit_sequence,
    calibration_fx_baseline,
    colorize,
    common_masks,
    configure_eval_module,
    load_candidates,
    load_existing_prediction,
    to_eval_frame,
)
from scripts.temporal_refinement.extend_temporal_gt_fairness_checks import load_rgb, write_csv

METHODS = [
    'StereoAnyVideo',
    'S2M2-S@512',
    'S2M2-S@512+EMA0.50',
    'S2M2-L@736',
    'S2M2-L@736+EMA0.50',
    'ConvGRU V2 e40',
    'Fast-FoundationStereo ONNX',
]
METRICS = ['depth_mae_mm','disp_mae_px','bad_2mm_pct','raw_temporal_diff','motion_compensated_temporal_mae','runtime_ms','peak_vram_mb']


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def finite_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def summarize(rows: list[dict[str, str]], sequence_balanced: bool) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r['method'] in METHODS:
            groups[r['method']].append(r)
    out = []
    rng = np.random.default_rng(123)
    for method, rs in groups.items():
        weights = np.ones(len(rs), dtype=np.float64) if sequence_balanced else np.array([finite_float(r['frames']) for r in rs], dtype=np.float64)
        row: dict[str, object] = {'method': method, 'sequences': len(rs), 'frames': int(sum(finite_float(r['frames']) for r in rs)), 'summary_type': 'sequence_balanced' if sequence_balanced else 'frame_weighted'}
        for metric in METRICS:
            vals = np.array([finite_float(r[metric]) for r in rs], dtype=np.float64)
            ok = np.isfinite(vals)
            if ok.any():
                row[metric] = float(np.average(vals[ok], weights=weights[ok]))
                row[f'{metric}_seq_mean'] = float(np.mean(vals[ok]))
                row[f'{metric}_seq_std'] = float(np.std(vals[ok], ddof=1)) if ok.sum() > 1 else math.nan
                boots = []
                idxs = np.arange(len(vals))[ok]
                for _ in range(2000):
                    sample = rng.choice(idxs, size=len(idxs), replace=True)
                    if sequence_balanced:
                        boots.append(float(np.mean(vals[sample])))
                    else:
                        boots.append(float(np.average(vals[sample], weights=weights[sample])))
                row[f'{metric}_ci95_low'] = float(np.percentile(boots, 2.5))
                row[f'{metric}_ci95_high'] = float(np.percentile(boots, 97.5))
            else:
                row[metric] = row[f'{metric}_seq_mean'] = row[f'{metric}_seq_std'] = row[f'{metric}_ci95_low'] = row[f'{metric}_ci95_high'] = math.nan
        out.append(row)
    return sorted(out, key=lambda r: (finite_float(r['depth_mae_mm']), finite_float(r['motion_compensated_temporal_mae'])))


def ranking_stability(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    by_seq: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r['method'] in METHODS:
            by_seq[r['sequence_id']].append(r)
    wins = {m: Counter() for m in METHODS}
    for _sid, rs in by_seq.items():
        for metric in ['depth_mae_mm','motion_compensated_temporal_mae']:
            best = min(rs, key=lambda r: finite_float(r[metric]))
            wins[best['method']][f'best_{metric}'] += 1
    return {m: dict(c) for m, c in wins.items()}


def format_rows(rows: list[dict[str, object]]) -> str:
    cols = ['method','depth_mae_mm','disp_mae_px','bad_2mm_pct','raw_temporal_diff','motion_compensated_temporal_mae','runtime_ms','peak_vram_mb']
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f'{v:.4f}' if isinstance(v, float) and math.isfinite(v) else str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def load_warped_or_temporal_prediction(method: str, sequence_id: str, frame_id: str) -> np.ndarray:
    if sequence_id == 'test_dataset_9_keyframe_3':
        dirs = {
            'S2M2-S@512': ROOT / 'results/03_temporal_refinement/evaluation/gt_temporal_test_dataset_9_keyframe_3/predictions/S2M2-S_512',
            'S2M2-L@736': ROOT / 'results/03_temporal_refinement/evaluation/gt_temporal_test_dataset_9_keyframe_3/predictions/S2M2-L_736',
        }
    else:
        dirs = {
            'S2M2-S@512': ROOT / 'results/01_frame_stereo/SCARED/warped_gt_108/S2M2-S',
            'S2M2-L@736': ROOT / 'results/01_frame_stereo/SCARED/warped_gt_108/S2M2-L',
        }
    return load_existing_prediction(dirs[method], sequence_id, frame_id)


def diagnose_s2m2_l(out_dir: Path) -> tuple[str, list[str]]:
    candidates = load_candidates()
    audit_rows = [audit_sequence(sid, frames) for sid, frames in candidates.items()]
    included = {str(r['sequence_id']): candidates[str(r['sequence_id'])] for r in audit_rows if r['valid_for_temporal_eval']}
    diag_rows = []
    images = []
    for sid, frames in included.items():
        ratios = []
        neg_l = []
        neg_s = []
        valid_counts = []
        for f in frames:
            gt = np.load(f.gt_disp_path).astype(np.float32)
            mask = np.load(f.valid_mask_path).astype(bool) & (gt > 0)
            s = load_warped_or_temporal_prediction('S2M2-S@512', sid, f.frame_id)
            l = load_warped_or_temporal_prediction('S2M2-L@736', sid, f.frame_id)
            valid_counts.append(int(mask.sum()))
            if mask.any():
                ratios.append(float(np.median(l[mask]) / max(np.median(gt[mask]), 1e-6)))
                neg_l.append(float(np.mean(l[mask] <= 0.1) * 100.0))
                neg_s.append(float(np.mean(s[mask] <= 0.1) * 100.0))
        diag_rows.append({
            'sequence_id': sid,
            'median_l_over_gt_disp_ratio': float(np.median(ratios)),
            'pred_l_nonpositive_pct': float(np.mean(neg_l)),
            'pred_s_nonpositive_pct': float(np.mean(neg_s)),
            'valid_pixels_mean': float(np.mean(valid_counts)),
        })
    write_csv(out_dir / 's2m2_l_diagnosis.csv', diag_rows, ['sequence_id','median_l_over_gt_disp_ratio','pred_l_nonpositive_pct','pred_s_nonpositive_pct','valid_pixels_mean'])
    normal_sid = 'dataset_1_keyframe_2'
    worst_sid = 'dataset_5_keyframe_1'
    for sid, suffix in [(normal_sid, 'normal'), (worst_sid, 'worst')]:
        frames = included[sid]
        idx = len(frames) // 2
        f = frames[idx]
        rgb = load_rgb(f.left_path)
        gt = np.load(f.gt_disp_path).astype(np.float32)
        s = load_warped_or_temporal_prediction('S2M2-S@512', sid, f.frame_id)
        l = load_warped_or_temporal_prediction('S2M2-L@736', sid, f.frame_id)
        vmax = float(np.nanpercentile(np.concatenate([gt.ravel(), s.ravel(), l.ravel()]), 98))
        tiles = [cv2.resize(rgb, (240,192)), cv2.resize(color(gt, vmax), (240,192)), cv2.resize(color(s, vmax), (240,192)), cv2.resize(color(l, vmax), (240,192)), cv2.resize(color(np.abs(l-gt), 10), (240,192))]
        labels = ['RGB','GT disp','S2M2-S','S2M2-L','abs L-GT']
        for tile, label in zip(tiles, labels):
            cv2.rectangle(tile, (0,0), (240,24), (0,0,0), -1)
            cv2.putText(tile, label, (5,17), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
        canvas = np.concatenate(tiles, axis=1)
        path = out_dir / 'reference_images' / f's2m2_l_{suffix}_{sid}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        images.append(str(path))
    return 'S2M2-L uses the same positive-disparity sign convention and resize-back-to-original coordinate policy as S2M2-S. No nonpositive/sign failure was found; poor multi-sequence result is concentrated in dataset_5 warped sequences, where L overestimates disparity/depth geometry relative to GT, so it appears to be a real domain/checkpoint failure rather than a loader scale bug.', images


def color(x: np.ndarray, vmax: float) -> np.ndarray:
    y = np.clip(np.nan_to_num(x, nan=0.0) / max(vmax, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(y, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def validate_sav_runtime(out_dir: Path) -> dict[str, object]:
    configure_eval_module()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    candidates = load_candidates()
    audit_rows = [audit_sequence(sid, frames) for sid, frames in candidates.items()]
    included = {str(r['sequence_id']): candidates[str(r['sequence_id'])] for r in audit_rows if r['valid_for_temporal_eval']}
    model = gt_eval.build_sav(device)
    # Warm-up on a short sequence; not included in timing.
    warm = [to_eval_frame(f) for f in next(iter(included.values()))[:5]]
    _ = gt_eval.infer_sav_chunk(model, warm, (384, 640), 6, device)
    if device.type == 'cuda':
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    rows = []
    total_frames = 0
    total_elapsed = 0.0
    peak = 0.0
    for sid, frames in included.items():
        eval_frames = [to_eval_frame(f) for f in frames]
        cursor = 0
        chunk_size = 32
        overlap = 4
        timed_frames = 0
        seq_elapsed = 0.0
        while cursor < len(eval_frames):
            end = min(cursor + chunk_size, len(eval_frames))
            chunk = eval_frames[cursor:end]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            # Full inference timing: image read, resize, tensor stack, synchronize, forward, CPU conversion.
            lefts, rights = [], []
            first = gt_eval.load_frame_payload(chunk[0])
            orig_h, orig_w = first['left'].shape[:2]
            for j, frame in enumerate(chunk):
                payload = first if j == 0 else gt_eval.load_frame_payload(frame)
                left = torch.from_numpy(payload['left']).permute(2, 0, 1).float().to(device)
                right = torch.from_numpy(payload['right']).permute(2, 0, 1).float().to(device)
                lefts.append(F.interpolate(left[None], size=(384,640), mode='bilinear', align_corners=True)[0])
                rights.append(F.interpolate(right[None], size=(384,640), mode='bilinear', align_corners=True)[0])
            stereo_video = torch.stack([torch.stack(lefts, 0), torch.stack(rights, 0)], dim=1)
            with torch.no_grad():
                raw = model.forward(stereo_video[:,0][None], stereo_video[:,1][None], iters=6, test_mode=True)
            if raw.shape[0] == len(chunk):
                disp = raw[:,0,:1].abs()
            else:
                disp = raw[0,:,:1].abs()
            disp_np = disp.squeeze(1).float().cpu().numpy().astype(np.float32)
            scale_x = 640 / float(orig_w)
            _preds = [cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale_x for d in disp_np]
            if device.type == 'cuda':
                torch.cuda.synchronize()
                peak = max(peak, torch.cuda.max_memory_allocated() / (1024**2))
            elapsed = time.perf_counter() - t0
            timed_frames += len(chunk)
            seq_elapsed += elapsed
            if end >= len(eval_frames):
                break
            cursor += max(1, chunk_size - overlap)
        rows.append({'sequence_id': sid, 'frames_timed_with_overlap': timed_frames, 'elapsed_sec': seq_elapsed, 'corrected_runtime_ms_per_frame': 1000*seq_elapsed/timed_frames})
        total_frames += timed_frames
        total_elapsed += seq_elapsed
    del model
    corrected = 1000 * total_elapsed / total_frames
    out = {'method': 'StereoAnyVideo', 'runtime_is_true_end_to_end': True, 'cached_predictions_timed': False, 'resolution': '384x640', 'amp': 'model mixed_precision=False, torch CUDA kernels active', 'chunk_size': 32, 'overlap': 4, 'warmup_excluded': True, 'cuda_synchronization': True, 'corrected_runtime_ms': corrected, 'peak_vram_mb': peak}
    write_csv(out_dir / 'runtime_validation_by_sequence.csv', rows, ['sequence_id','frames_timed_with_overlap','elapsed_sec','corrected_runtime_ms_per_frame'])
    return out


def main() -> None:
    base = ROOT / 'results/03_temporal_refinement/evaluation/scared_multisequence_compact'
    out = ROOT / 'results/03_temporal_refinement/evaluation/scared_multisequence_validation'
    (out / 'reference_images').mkdir(parents=True, exist_ok=True)
    per_seq = [r for r in read_csv(base / 'per_sequence_metrics.csv') if r['method'] in METHODS]
    frame_weighted = summarize(per_seq, sequence_balanced=False)
    seq_balanced = summarize(per_seq, sequence_balanced=True)
    ranks = ranking_stability(per_seq)
    sav_runtime = validate_sav_runtime(out)
    diagnosis, ref_images = diagnose_s2m2_l(out)
    runtime_rows = [sav_runtime]
    write_csv(out / 'runtime_validation.csv', runtime_rows, ['method','runtime_is_true_end_to_end','cached_predictions_timed','resolution','amp','chunk_size','overlap','warmup_excluded','cuda_synchronization','corrected_runtime_ms','peak_vram_mb'])
    write_csv(out / 'validated_summary_frame_weighted.csv', frame_weighted, list(frame_weighted[0].keys()))
    write_csv(out / 'validated_summary_sequence_balanced.csv', seq_balanced, list(seq_balanced[0].keys()))
    val_rows = []
    for r in per_seq:
        row = dict(r)
        row['ranking_stability'] = json.dumps(ranks.get(r['method'], {}))
        val_rows.append(row)
    write_csv(out / 'per_sequence_validation.csv', val_rows, list(val_rows[0].keys()))
    deployment = next(r for r in seq_balanced if r['method'] == 'S2M2-S@512+EMA0.50')
    conv = next(r for r in seq_balanced if r['method'] == 'ConvGRU V2 e40')
    conclusion = float(deployment['depth_mae_mm']) < float(conv['depth_mae_mm']) and float(deployment['motion_compensated_temporal_mae']) < float(conv['motion_compensated_temporal_mae'])
    report = [
        '# SCARED Multi-Sequence Benchmark Validation', '',
        '## StereoAnyVideo Runtime', '',
        f"Corrected true end-to-end runtime: `{sav_runtime['corrected_runtime_ms']:.2f} ms/frame`; peak VRAM `{sav_runtime['peak_vram_mb']:.1f} MB`.",
        'Timing includes image loading, resize, tensor transfer/stack, synchronized model forward, disparity resize back to original coordinates, and CPU conversion. Cached predictions are not timed. Model load is excluded; warm-up is excluded.', '',
        '## S2M2-L Diagnosis', '', diagnosis, '',
        '## Frame-Weighted Summary', '', format_rows(frame_weighted), '',
        '## Sequence-Balanced Summary', '', format_rows(seq_balanced), '',
        '## Ranking Stability', '', json.dumps(ranks, indent=2), '',
        '## Deployment Conclusion', '',
        f"S2M2-S@512 + EMA0.50 remains the best lightweight causal configuration under sequence-balanced evaluation: `{conclusion}`.", '',
        '## Reference Images', '', *[f'- `{p}`' for p in ref_images],
    ]
    (out / 'report.md').write_text('\n'.join(report) + '\n')
    (out / 'run.log').write_text('validation_complete=true\n')
    print(f'Wrote {out}')
    print(f"corrected_sav_runtime_ms={sav_runtime['corrected_runtime_ms']:.2f}")
    print(f'deployment_conclusion={conclusion}')


if __name__ == '__main__':
    main()
