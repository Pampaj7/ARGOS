import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
S2M2_SRC = REPO_ROOT / "stereo/s2m2/src"
sys.path.insert(0, str(S2M2_SRC))

from s2m2.core.model.s2m2 import S2M2
from s2m2.core.utils.image_utils import image_crop, image_pad
from eval_metrics import failure_aware_metrics, summarize_rows


MODEL_CONFIG = {
    "S": {"feature_channels": 128, "n_transformer": 1},
    "M": {"feature_channels": 192, "n_transformer": 2},
    "L": {"feature_channels": 256, "n_transformer": 3},
    "XL": {"feature_channels": 384, "n_transformer": 3},
}


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_gt(ref_dir, stem):
    disp_npy = ref_dir / "Disparity_float32" / f"{stem}.npy"
    depth_npy = ref_dir / "DepthL_float32" / f"{stem}.npy"
    mask_npy = ref_dir / "ValidMask" / f"{stem}.npy"
    if disp_npy.exists() and depth_npy.exists():
        gt = np.load(disp_npy).astype(np.float32)
        gt_depth = np.load(depth_npy).astype(np.float32)
        if mask_npy.exists():
            gt_mask = np.load(mask_npy).astype(bool)
        else:
            gt_mask = (gt > 0) & (gt_depth > 0) & np.isfinite(gt) & np.isfinite(gt_depth)
        return gt, gt_depth, gt_mask, "float32_npy"

    gt = cv2.imread(str(ref_dir / "Disparity" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
    gt_depth = cv2.imread(str(ref_dir / "DepthL" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
    gt_mask = (gt > 0) & (gt_depth > 0) & np.isfinite(gt) & np.isfinite(gt_depth)
    return gt, gt_depth, gt_mask, "uint16_png"


def build_model(model_type, refine_iter, device):
    cfg = MODEL_CONFIG[model_type]
    model = S2M2(
        feature_channels=cfg["feature_channels"],
        dim_expansion=1,
        num_transformer=cfg["n_transformer"],
        use_positivity=True,
        refine_iter=refine_iter,
    )
    return model.to(device).eval()


def load_pretrained(model, weights_dir, model_type):
    cfg = MODEL_CONFIG[model_type]
    ckpt_path = Path(weights_dir) / f"CH{cfg['feature_channels']}NTR{cfg['n_transformer']}.pth"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.my_load_state_dict(checkpoint["state_dict"])


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=True)


def disp_vis(x, max_val):
    x = np.clip(x.astype(np.float32), 0, max(max_val, 1e-6))
    x = (x / max(max_val, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)[..., ::-1]


def maybe_resize_pair(left, right, max_width):
    h, w = left.shape[:2]
    if not max_width or w <= max_width:
        return left, right, 1.0
    scale = max_width / float(w)
    new_size = (max_width, int(round(h * scale)))
    left = cv2.resize(left, new_size, interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(right, new_size, interpolation=cv2.INTER_LINEAR)
    return left, right, scale


@torch.no_grad()
def infer(model, left, right, device, max_width):
    orig_h, orig_w = left.shape[:2]
    left_in, right_in, scale = maybe_resize_pair(left, right, max_width)
    left_t = torch.from_numpy(left_in).permute(2, 0, 1).unsqueeze(0).float().to(device)
    right_t = torch.from_numpy(right_in).permute(2, 0, 1).unsqueeze(0).float().to(device)
    h, w = left_t.shape[-2:]
    left_p = image_pad(left_t, 32)
    right_p = image_pad(right_t, 32)
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        pred, _occ, _conf = model(left_p, right_p)
    pred = image_crop(pred, (h, w)).squeeze().float().cpu().numpy()
    if scale != 1.0:
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale
    return np.clip(pred, 0, None).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared_root", default="stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes")
    parser.add_argument("--weights_dir", default="stereo/s2m2/weights/pretrain_weights")
    parser.add_argument("--model_type", default="L", choices=["S", "M", "L", "XL"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--refine_iter", type=int, default=3)
    parser.add_argument("--max_width", type=int, default=1024, help="Resize wide inputs before inference; 0 keeps full resolution")
    parser.add_argument("--out_dir", default="results/scared_s2m2_L_eval")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    model = build_model(args.model_type, args.refine_iter, device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    else:
        load_pretrained(model, args.weights_dir, args.model_type)

    rows = []
    montage_rows = []
    root = Path(args.scared_root)
    for exp_dir in sorted(root.glob("dataset_*")):
        if args.datasets and exp_dir.name not in set(args.datasets):
            continue
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
            pred = infer(model, left, right, device, args.max_width)
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
                max_disp = np.percentile(gt[gt > 0], 99)
                err = np.abs(pred - gt)
                err_max = min(50, np.percentile(err[mask], 99))
                triptych = np.concatenate(
                    [
                        left,
                        disp_vis(pred, max_disp),
                        disp_vis(gt, max_disp),
                        disp_vis(err, err_max),
                    ],
                    axis=1,
                )
                montage_rows.append((f"{exp_dir.name}/{stem}", triptych))
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
        thumbs = []
        for label, triptych in montage_rows:
            thumb = cv2.resize(triptych, (1440, 288), interpolation=cv2.INTER_AREA)
            canvas = np.full((312, 1440, 3), 255, dtype=np.uint8)
            cv2.putText(canvas, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            canvas[24:] = thumb
            thumbs.append(canvas)
        cv2.imwrite(str(out_dir / "montage_left_pred_gt_err.png"), np.concatenate(thumbs, axis=0)[..., ::-1])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
