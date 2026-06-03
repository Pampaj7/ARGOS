import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import yaml

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from run_demo_single_trt import OnnxRuntimeRunner, normalize_imagenet
from Utils import set_logging_format, set_seed, vis_disparity


def load_image(path):
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    return img[..., :3]


def metrics(pred, gt, mask):
    err = np.abs(pred[mask] - gt[mask])
    return {
        'valid_px': int(mask.sum()),
        'mae_px': float(err.mean()),
        'rmse_px': float(np.sqrt((err ** 2).mean())),
        'bad1_pct': float((err > 1.0).mean() * 100.0),
        'bad2_pct': float((err > 2.0).mean() * 100.0),
        'bad5_pct': float((err > 5.0).mean() * 100.0),
    }


def depth_metrics(pred_mm, gt_mm, mask):
    err = np.abs(pred_mm[mask] - gt_mm[mask])
    return {
        'depth_mae_mm': float(err.mean()),
        'depth_rmse_mm': float(np.sqrt((err ** 2).mean())),
        'depth_bad1mm_pct': float((err > 1.0).mean() * 100.0),
        'depth_bad2mm_pct': float((err > 2.0).mean() * 100.0),
        'depth_bad5mm_pct': float((err > 5.0).mean() * 100.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--servct_root', default='data/surgical_stereo/servct/SERV-CT')
    parser.add_argument('--model_file', default='weights/onnx/20_30_48/320x736/20_30_48_iters_4_res_320x736.onnx')
    parser.add_argument('--out_dir', default='output_servct_eval')
    parser.add_argument('--reference', choices=['Reference_CT', 'Reference_RGB'], default='Reference_CT')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.model_file).with_suffix('.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    target_h, target_w = cfg['image_size']
    runner = OnnxRuntimeRunner(args.model_file)

    root = Path(args.servct_root)
    rows = []
    montage_rows = []
    for exp in ['Experiment_1', 'Experiment_2']:
        exp_dir = root / exp
        ref_dir = exp_dir / args.reference
        if not ref_dir.exists():
            continue
        for left_path in sorted((exp_dir / 'Left_rectified').glob('*.png')):
            stem = left_path.stem
            right_path = exp_dir / 'Right_rectified' / f'{stem}.png'
            gt_path = ref_dir / 'Disparity' / f'{stem}.png'
            gt_depth_path = ref_dir / 'DepthL' / f'{stem}.png'
            calib_path = exp_dir / 'Rectified_calibration' / f'{stem}.json'
            if not right_path.exists() or not gt_path.exists():
                continue

            left = load_image(left_path)
            right = load_image(right_path)
            orig_h, orig_w = left.shape[:2]
            sx = target_w / orig_w

            left_res = cv2.resize(left, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            right_res = cv2.resize(right, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            t_left = torch.as_tensor(normalize_imagenet(left_res)).cuda().float()[None].permute(0, 3, 1, 2)
            t_right = torch.as_tensor(normalize_imagenet(right_res)).cuda().float()[None].permute(0, 3, 1, 2)

            pred = runner({'left_image': t_left, 'right_image': t_right})['disparity']
            pred = pred.float().cpu().numpy().reshape(target_h, target_w).clip(0, None)

            gt_orig = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            gt = cv2.resize(gt_orig, (target_w, target_h), interpolation=cv2.INTER_LINEAR) * sx
            gt_depth_orig = cv2.imread(str(gt_depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            gt_depth = cv2.resize(gt_depth_orig, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            calib = json.loads(calib_path.read_text())
            p1 = np.array(calib['P1']['data'], dtype=np.float32).reshape(3, 4)
            p2 = np.array(calib['P2']['data'], dtype=np.float32).reshape(3, 4)
            f_target = p1[0, 0] * sx
            baseline_mm = abs(p2[0, 3] / p2[0, 0])
            pred_depth = f_target * baseline_mm / np.maximum(pred, 1e-6)
            mask = (gt > 0) & (gt_depth > 0) & np.isfinite(pred_depth)
            row = {'experiment': exp, 'frame': stem}
            row.update(metrics(pred, gt, mask))
            row.update(depth_metrics(pred_depth, gt_depth, mask))
            rows.append(row)

            max_disp = np.percentile(gt[mask], 99)
            pred_vis = vis_disparity(pred, min_val=0, max_val=max_disp, invalid_thres=np.inf)
            gt_vis = vis_disparity(gt, min_val=0, max_val=max_disp, invalid_thres=np.inf)
            err = np.abs(pred - gt)
            err_vis = vis_disparity(err, min_val=0, max_val=min(20, np.percentile(err[mask], 99)))
            triptych = np.concatenate([left_res, pred_vis, gt_vis, err_vis], axis=1)
            cv2.imwrite(str(Path(args.out_dir) / f'{exp}_{stem}_left_pred_gt_err.png'), triptych[..., ::-1])
            montage_rows.append((f'{exp}/{stem}', triptych))

    if not rows:
        raise RuntimeError('No SERV-CT frames found')

    keys = [
        'experiment', 'frame', 'valid_px',
        'mae_px', 'rmse_px', 'bad1_pct', 'bad2_pct', 'bad5_pct',
        'depth_mae_mm', 'depth_rmse_mm', 'depth_bad1mm_pct',
        'depth_bad2mm_pct', 'depth_bad5mm_pct',
    ]
    with open(Path(args.out_dir) / 'metrics.csv', 'w') as f:
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join(str(row[k]) for k in keys) + '\n')

    numeric_keys = keys[2:]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in numeric_keys}
    summary['frames'] = len(rows)
    with open(Path(args.out_dir) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    thumbs = []
    for label, img in montage_rows:
        thumb = cv2.resize(img, (target_w * 2, target_h // 2), interpolation=cv2.INTER_AREA)
        canvas = np.full((thumb.shape[0] + 24, thumb.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(canvas, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[24:] = thumb
        thumbs.append(canvas)
    montage = np.concatenate(thumbs, axis=0)
    cv2.imwrite(str(Path(args.out_dir) / 'montage_left_pred_gt_err.png'), montage[..., ::-1])
    logging.info(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
