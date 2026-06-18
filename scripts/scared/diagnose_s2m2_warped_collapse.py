import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stereo/s2m2/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_scared_s2m2 import build_model, infer, load_checkpoint, load_pretrained, read_rgb
from eval_metrics import failure_aware_metrics, summarize_rows


THRESHOLDS = [1, 2, 3, 5, 10]


def load_model_pair(args, device):
    zero = build_model("XL", 3, device)
    load_pretrained(zero, args.weights_dir, "XL")
    ft = build_model("XL", 3, device)
    load_checkpoint(ft, args.ft_checkpoint)
    return zero.eval(), ft.eval()


def depth_from_disp(pred, calib_path):
    calib = json.loads(Path(calib_path).read_text())
    p1 = np.array(calib["P1"]["data"], dtype=np.float32).reshape(3, 4)
    p2 = np.array(calib["P2"]["data"], dtype=np.float32).reshape(3, 4)
    return float(p1[0, 0]) * abs(float(p2[0, 3] / p2[0, 0])) / np.maximum(pred, 1e-6)


def clean_samples(root, datasets):
    root = Path(root)
    rows = []
    for dataset in datasets:
        exp_dir = root / dataset
        ref = exp_dir / "Reference_SCARED"
        for left in sorted((exp_dir / "Left_rectified").glob("*.png")):
            stem = left.stem
            rows.append(
                {
                    "split": "clean",
                    "dataset": dataset,
                    "frame": stem,
                    "left": str(left),
                    "right": str(exp_dir / "Right_rectified" / f"{stem}.png"),
                    "disp": str(ref / "Disparity_float32" / f"{stem}.npy"),
                    "depth": str(ref / "DepthL_float32" / f"{stem}.npy"),
                    "mask": str(ref / "ValidMask" / f"{stem}.npy"),
                    "calib": str(exp_dir / "Rectified_calibration" / f"{stem}.json"),
                }
            )
    return rows


def warped_samples(metadata_csv, limit, seed):
    rows = [
        r
        for r in csv.DictReader(open(metadata_csv))
        if r["notes"] == "" and float(r["valid_pixel_ratio"]) >= 0.2 and Path(r["left_path"]).exists()
    ]
    random.Random(seed).shuffle(rows)
    rows = rows[:limit]
    out = []
    for r in rows:
        out.append(
            {
                "split": "warped_train",
                "dataset": r["dataset_id"],
                "frame": f"{r['keyframe_id']}/frame_{int(r['frame_id']):06d}",
                "left": r["left_path"],
                "right": r["right_path"],
                "disp": r["disparity_float32_path"],
                "depth": r["depth_float32_path"],
                "mask": r["valid_mask_path"],
                "calib": r["calibration_path"],
            }
        )
    return out


def metric_row(sample, model_name, pred, gt, gt_depth, gt_mask, pred_depth):
    mask = gt_mask & np.isfinite(pred_depth)
    signed = pred[mask] - gt[mask]
    row = {"split": sample["split"], "dataset": sample["dataset"], "frame": sample["frame"], "model": model_name}
    row.update(failure_aware_metrics(pred, pred_depth, gt, gt_depth, gt_mask, mask))
    row.update(
        {
            "signed_disp_error_mean": float(signed.mean()),
            "signed_disp_error_median": float(np.median(signed)),
            "pred_disp_p05": float(np.percentile(pred[mask], 5)),
            "pred_disp_p50": float(np.percentile(pred[mask], 50)),
            "pred_disp_p95": float(np.percentile(pred[mask], 95)),
            "gt_disp_p05": float(np.percentile(gt[mask], 5)),
            "gt_disp_p50": float(np.percentile(gt[mask], 50)),
            "gt_disp_p95": float(np.percentile(gt[mask], 95)),
        }
    )
    return row, signed


def disp_panel(x, max_val):
    x = np.clip(x.astype(np.float32), 0, max(max_val, 1e-6))
    x = (x / max(max_val, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)[..., ::-1]


def signed_panel(x, max_abs=30):
    x = np.clip(x.astype(np.float32), -max_abs, max_abs)
    y = ((x + max_abs) / (2 * max_abs) * 255).astype(np.uint8)
    return cv2.applyColorMap(y, cv2.COLORMAP_COOL)[..., ::-1]


def make_montage(items, out_path):
    rows = []
    for item in items[:30]:
        left, gt, zero, ft, mask, label = item
        max_disp = np.percentile(gt[mask], 99)
        panels = [
            left,
            disp_panel(gt, max_disp),
            disp_panel(zero, max_disp),
            disp_panel(ft, max_disp),
            signed_panel(zero - gt),
            signed_panel(ft - gt),
        ]
        panels = [cv2.resize(p, (220, 176), interpolation=cv2.INTER_AREA) for p in panels]
        strip = np.concatenate(panels, axis=1)
        canvas = np.full((200, strip.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(canvas, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[24:] = strip
        rows.append(canvas)
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0)[..., ::-1])


def plot_hist(errors, out_path, title):
    plt.figure(figsize=(7, 4))
    for name, vals in errors.items():
        if vals:
            x = np.concatenate(vals)
            x = x[np.isfinite(x)]
            if x.size > 400000:
                x = np.random.default_rng(7).choice(x, 400000, replace=False)
            plt.hist(x, bins=120, range=(-40, 40), alpha=0.5, label=name, density=True)
    plt.xlabel("signed disparity error pred - GT (px)")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def evaluate_samples(samples, zero, ft, device, max_width, out_dir):
    rows = []
    signed_by_split = {}
    montage_by_split = {}
    for sample in samples:
        left = read_rgb(sample["left"])
        right = read_rgb(sample["right"])
        gt = np.load(sample["disp"]).astype(np.float32)
        gt_depth = np.load(sample["depth"]).astype(np.float32)
        gt_mask = np.load(sample["mask"]).astype(bool)
        pred_zero = infer(zero, left, right, device, max_width)
        pred_ft = infer(ft, left, right, device, max_width)
        for name, pred in [("zero_shot", pred_zero), ("ft_step500", pred_ft)]:
            pred_depth = depth_from_disp(pred, sample["calib"])
            row, signed = metric_row(sample, name, pred, gt, gt_depth, gt_mask, pred_depth)
            rows.append(row)
            signed_by_split.setdefault(f"{sample['split']}_{name}", []).append(signed)
        montage_by_split.setdefault(sample["split"], []).append(
            (left, gt, pred_zero, pred_ft, gt_mask, f"{sample['dataset']}/{sample['frame']}")
        )
        print(f"{sample['split']} {sample['dataset']}/{sample['frame']}", flush=True)

    with open(out_dir / "per_frame_comparison.csv", "w", newline="") as f:
        keys = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for split in sorted({r["split"] for r in rows}):
        for model in ["zero_shot", "ft_step500"]:
            subset = [r for r in rows if r["split"] == split and r["model"] == model]
            summary = summarize_rows(subset, skip_keys=("split", "dataset", "frame", "model"))
            summary["split"] = split
            summary["model"] = model
            summary_rows.append(summary)
    with open(out_dir / "summary_comparison.csv", "w", newline="") as f:
        keys = ["split", "model"] + [k for k in summary_rows[0] if k not in ("split", "model")]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)
    (out_dir / "summary_comparison.json").write_text(json.dumps(summary_rows, indent=2))

    for split, items in montage_by_split.items():
        make_montage(items, out_dir / f"montage_{split}_zero_vs_ft.png")
    for split in sorted({s.rsplit("_", 1)[0] for s in signed_by_split}):
        plot_hist(
            {k.replace(split + "_", ""): v for k, v in signed_by_split.items() if k.startswith(split + "_")},
            out_dir / f"signed_disp_error_hist_{split}.png",
            split,
        )
    return summary_rows


def teacher_agreement(samples, zero, device, max_width, out_dir):
    mask_dir = out_dir / "teacher_agreement_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    totals = {t: 0 for t in THRESHOLDS}
    valid_total = 0
    rows = []
    for sample in samples:
        left = read_rgb(sample["left"])
        right = read_rgb(sample["right"])
        gt = np.load(sample["disp"]).astype(np.float32)
        mask = np.load(sample["mask"]).astype(bool)
        pred = infer(zero, left, right, device, max_width)
        err = np.abs(pred - gt)
        valid = mask & np.isfinite(err)
        valid_count = int(valid.sum())
        valid_total += valid_count
        bitmask = np.zeros(mask.shape, dtype=np.uint8)
        row = {"dataset": sample["dataset"], "frame": sample["frame"], "valid_px": valid_count}
        for bit, t in enumerate(THRESHOLDS):
            keep = valid & (err <= t)
            bitmask[keep] |= np.uint8(1 << bit)
            count = int(keep.sum())
            totals[t] += count
            row[f"retained_le_{t}px"] = count
            row[f"retained_le_{t}px_ratio"] = count / max(valid_count, 1)
        safe = f"{sample['dataset']}_{sample['frame'].replace('/', '_')}"
        np.savez_compressed(mask_dir / f"{safe}_agreement_bitmask.npz", bitmask=bitmask, thresholds=np.array(THRESHOLDS))
        rows.append(row)
        print(f"agreement {sample['dataset']}/{sample['frame']}", flush=True)
    with open(out_dir / "teacher_gt_agreement_per_frame.csv", "w", newline="") as f:
        keys = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {f"retention_le_{t}px": totals[t] / max(valid_total, 1) for t in THRESHOLDS}
    summary["frames"] = len(rows)
    summary["valid_px"] = valid_total
    (out_dir / "teacher_gt_agreement_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", default="stereo/s2m2/weights/pretrain_weights")
    parser.add_argument("--ft_checkpoint", default="results/01_frame_stereo/SCARED/scared_s2m2_xl_warped_finetune_run1/checkpoints/step_000500.pth")
    parser.add_argument("--clean_root", default="stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes")
    parser.add_argument("--warped_metadata", default="results/scared_warped_train_subset_metadata.csv")
    parser.add_argument("--out_dir", default="results/01_frame_stereo/SCARED/scared_s2m2_xl_warped_collapse_diagnosis")
    parser.add_argument("--warped_compare_limit", type=int, default=120)
    parser.add_argument("--agreement_limit", type=int, default=300)
    parser.add_argument("--max_width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    zero, ft = load_model_pair(args, device)
    samples = (
        clean_samples(args.clean_root, ["dataset_7"])
        + clean_samples(args.clean_root, ["dataset_8", "dataset_9"])
        + warped_samples(args.warped_metadata, args.warped_compare_limit, args.seed)
    )
    summary = evaluate_samples(samples, zero, ft, device, args.max_width, out_dir)
    agreement_samples = warped_samples(args.warped_metadata, args.agreement_limit, args.seed)
    agreement = teacher_agreement(agreement_samples, zero, device, args.max_width, out_dir)
    print(json.dumps({"comparison": summary, "agreement": agreement}, indent=2))


if __name__ == "__main__":
    main()
