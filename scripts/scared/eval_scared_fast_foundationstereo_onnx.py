import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from eval_metrics import failure_aware_metrics, summarize_rows
from eval_scared_s2m2 import disp_vis, load_gt, read_rgb


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_imagenet(img_rgb):
    return ((img_rgb.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD


def make_session(model_file):
    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(model_file), providers=providers)
    return session


def infer(session, left, right, target_h, target_w):
    orig_h, orig_w = left.shape[:2]
    sx = target_w / float(orig_w)
    left_res = cv2.resize(left, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    right_res = cv2.resize(right, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    left_t = normalize_imagenet(left_res).transpose(2, 0, 1)[None].astype(np.float32)
    right_t = normalize_imagenet(right_res).transpose(2, 0, 1)[None].astype(np.float32)
    outputs = session.run(None, {"left_image": left_t, "right_image": right_t})
    pred = outputs[0].reshape(target_h, target_w).astype(np.float32).clip(0, None)
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / sx
    return pred.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared_root", default="stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes")
    parser.add_argument(
        "--model_file",
        default="stereo/Fast-FoundationStereo/weights/onnx/20_30_48/320x736/20_30_48_iters_4_res_320x736.onnx",
    )
    parser.add_argument("--out_dir", default="results/scared_fast_foundationstereo_onnx_eval")
    args = parser.parse_args()

    model_file = Path(args.model_file)
    cfg = yaml.safe_load(model_file.with_suffix(".yaml").read_text())
    target_h, target_w = cfg["image_size"]
    session = make_session(model_file)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "scared_root": args.scared_root,
                "model_file": args.model_file,
                "image_size": [target_h, target_w],
                "onnxruntime_providers": session.get_providers(),
            },
            indent=2,
        )
    )

    rows = []
    montage_rows = []
    root = Path(args.scared_root)
    for exp_dir in sorted(root.glob("dataset_*")):
        ref_dir = exp_dir / "Reference_SCARED"
        for left_path in sorted((exp_dir / "Left_rectified").glob("*.png")):
            stem = left_path.stem
            right_path = exp_dir / "Right_rectified" / f"{stem}.png"
            calib_path = exp_dir / "Rectified_calibration" / f"{stem}.json"
            if not all(p.exists() for p in [right_path, calib_path]):
                continue

            left = read_rgb(left_path)
            right = read_rgb(right_path)
            t0 = time.perf_counter()
            pred = infer(session, left, right, target_h, target_w)
            runtime_ms = (time.perf_counter() - t0) * 1000.0

            gt, gt_depth, gt_mask, gt_source = load_gt(ref_dir, stem)
            calib = json.loads(calib_path.read_text())
            p1 = np.array(calib["P1"]["data"], dtype=np.float32).reshape(3, 4)
            p2 = np.array(calib["P2"]["data"], dtype=np.float32).reshape(3, 4)
            f = p1[0, 0]
            baseline_mm = abs(p2[0, 3] / p2[0, 0])
            pred_depth = f * baseline_mm / np.maximum(pred, 1e-6)
            mask = gt_mask & np.isfinite(pred_depth)

            row = {"dataset": exp_dir.name, "frame": stem, "gt_source": gt_source, "runtime_ms": runtime_ms}
            row.update(failure_aware_metrics(pred, pred_depth, gt, gt_depth, gt_mask, mask))
            rows.append(row)

            if len(montage_rows) < 18:
                max_disp = np.percentile(gt[gt_mask], 99)
                err = np.abs(pred - gt)
                err_max = min(50, np.percentile(err[mask], 99))
                triptych = np.concatenate(
                    [left, disp_vis(pred, max_disp), disp_vis(gt, max_disp), disp_vis(err, err_max)],
                    axis=1,
                )
                thumb = cv2.resize(triptych, (1440, 288), interpolation=cv2.INTER_AREA)
                canvas = np.full((312, 1440, 3), 255, dtype=np.uint8)
                cv2.putText(canvas, f"{exp_dir.name}/{stem}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
                canvas[24:] = thumb
                montage_rows.append(canvas)
            print(f"{exp_dir.name}/{stem}", flush=True)

    if not rows:
        raise RuntimeError("No SCARED converted frames found")

    keys = list(rows[0].keys())
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if montage_rows:
        cv2.imwrite(str(out_dir / "montage_left_pred_gt_err.png"), np.concatenate(montage_rows, axis=0)[..., ::-1])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
