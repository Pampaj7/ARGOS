import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from s2m2.core.model.s2m2 import S2M2
from s2m2.core.utils.image_utils import image_crop, image_pad
from s2m2.core.utils.vis_utils import apply_colormap


MODEL_CONFIG = {
    "S": {"feature_channels": 128, "n_transformer": 1},
    "M": {"feature_channels": 192, "n_transformer": 2},
    "L": {"feature_channels": 256, "n_transformer": 3},
    "XL": {"feature_channels": 384, "n_transformer": 3},
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_type, refine_iter, use_positivity):
    cfg = MODEL_CONFIG[model_type]
    return S2M2(
        feature_channels=cfg["feature_channels"],
        dim_expansion=1,
        num_transformer=cfg["n_transformer"],
        use_positivity=use_positivity,
        refine_iter=refine_iter,
    )


def load_pretrained(model, weights_dir, model_type):
    cfg = MODEL_CONFIG[model_type]
    ckpt_path = Path(weights_dir) / f"CH{cfg['feature_channels']}NTR{cfg['n_transformer']}.pth"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.my_load_state_dict(checkpoint["state_dict"])


def set_trainable(model, mode):
    for p in model.parameters():
        p.requires_grad = mode == "all"

    if mode == "refiners":
        modules = [
            model.global_refiner,
            model.feat_fusion_layer,
            model.refiner,
            model.ctx_feat,
            model.upsample_mask_4x_refine,
            model.upsample_mask_1x,
        ]
        for module in modules:
            for p in module.parameters():
                p.requires_grad = True


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class ServCtDataset(Dataset):
    def __init__(self, servct_root, samples, crop_size=None, augment=False):
        self.root = Path(servct_root)
        self.samples = samples
        self.crop_size = crop_size
        self.augment = augment

    @staticmethod
    def collect(servct_root, experiments, references):
        root = Path(servct_root)
        samples = []
        for exp in experiments:
            exp_dir = root / exp
            for ref in references:
                ref_dir = exp_dir / ref
                if not ref_dir.exists():
                    continue
                for left_path in sorted((exp_dir / "Left_rectified").glob("*.png")):
                    stem = left_path.stem
                    right_path = exp_dir / "Right_rectified" / f"{stem}.png"
                    disp_path = ref_dir / "Disparity" / f"{stem}.png"
                    depth_path = ref_dir / "DepthL" / f"{stem}.png"
                    if right_path.exists() and disp_path.exists() and depth_path.exists():
                        samples.append(
                            {
                                "experiment": exp,
                                "reference": ref,
                                "frame": stem,
                                "left": left_path,
                                "right": right_path,
                                "disp": disp_path,
                                "depth": depth_path,
                            }
                        )
        return samples

    def __len__(self):
        return len(self.samples)

    def _crop(self, left, right, disp, depth):
        if self.crop_size is None:
            return left, right, disp, depth
        crop_h, crop_w = self.crop_size
        h, w = disp.shape
        if h <= crop_h or w <= crop_w:
            return left, right, disp, depth
        y = random.randint(0, h - crop_h)
        x = random.randint(0, w - crop_w)
        return (
            left[y : y + crop_h, x : x + crop_w],
            right[y : y + crop_h, x : x + crop_w],
            disp[y : y + crop_h, x : x + crop_w],
            depth[y : y + crop_h, x : x + crop_w],
        )

    def _augment(self, left, right):
        if not self.augment:
            return left, right
        gain = random.uniform(0.85, 1.15)
        bias = random.uniform(-8, 8)
        gamma = random.uniform(0.9, 1.1)
        left_base = np.clip((left.astype(np.float32) * gain + bias) / 255.0, 0, 1)
        right_base = np.clip((right.astype(np.float32) * gain + bias) / 255.0, 0, 1)
        left = np.clip(left_base ** gamma * 255.0, 0, 255)
        right = np.clip(right_base ** gamma * 255.0, 0, 255)
        if random.random() < 0.25:
            noise = np.random.normal(0, 2.0, left.shape).astype(np.float32)
            left = np.clip(left + noise, 0, 255)
            right = np.clip(right + noise, 0, 255)
        return left.astype(np.uint8), right.astype(np.uint8)

    def __getitem__(self, idx):
        s = self.samples[idx]
        left = read_rgb(s["left"])
        right = read_rgb(s["right"])
        disp = cv2.imread(str(s["disp"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        depth = cv2.imread(str(s["depth"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        left, right, disp, depth = self._crop(left, right, disp, depth)
        left, right = self._augment(left, right)
        return {
            "left": torch.from_numpy(left.copy()).permute(2, 0, 1).float(),
            "right": torch.from_numpy(right.copy()).permute(2, 0, 1).float(),
            "disp": torch.from_numpy(disp.copy()).unsqueeze(0).float(),
            "depth": torch.from_numpy(depth.copy()).unsqueeze(0).float(),
            "experiment": s["experiment"],
            "reference": s["reference"],
            "frame": s["frame"],
        }


def pad_batch(left, right, disp=None, depth=None):
    h, w = left.shape[-2:]
    left_p = image_pad(left, 32)
    right_p = image_pad(right, 32)
    if disp is None:
        return left_p, right_p, (h, w), None, None
    pad_h = left_p.shape[-2] - h
    pad_w = left_p.shape[-1] - w
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return left_p, right_p, (h, w), F.pad(disp, pad, mode="replicate"), F.pad(depth, pad, mode="replicate")


def gradient_loss(pred, gt, mask):
    mask_x = mask[..., :, 1:] & mask[..., :, :-1]
    mask_y = mask[..., 1:, :] & mask[..., :-1, :]
    loss_x = (pred[..., :, 1:] - pred[..., :, :-1] - gt[..., :, 1:] + gt[..., :, :-1]).abs()[mask_x].mean()
    loss_y = (pred[..., 1:, :] - pred[..., :-1, :] - gt[..., 1:, :] + gt[..., :-1, :]).abs()[mask_y].mean()
    return loss_x + loss_y


def disp_metrics(pred, gt, mask):
    err = np.abs(pred[mask] - gt[mask])
    return {
        "valid_px": int(mask.sum()),
        "mae_px": float(err.mean()),
        "rmse_px": float(np.sqrt((err ** 2).mean())),
        "bad1_pct": float((err > 1.0).mean() * 100.0),
        "bad2_pct": float((err > 2.0).mean() * 100.0),
        "bad5_pct": float((err > 5.0).mean() * 100.0),
    }


def depth_metrics(pred_mm, gt_mm, mask):
    err = np.abs(pred_mm[mask] - gt_mm[mask])
    return {
        "depth_mae_mm": float(err.mean()),
        "depth_rmse_mm": float(np.sqrt((err ** 2).mean())),
        "depth_bad1mm_pct": float((err > 1.0).mean() * 100.0),
        "depth_bad2mm_pct": float((err > 2.0).mean() * 100.0),
        "depth_bad5mm_pct": float((err > 5.0).mean() * 100.0),
    }


def disp_vis(x, max_val):
    return apply_colormap(np.clip(x.astype(np.float32), 0, max_val))


@torch.no_grad()
def evaluate(model, samples, servct_root, out_dir, device):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    montage_rows = []
    model.eval()
    for s in samples:
        left = read_rgb(s["left"])
        right = read_rgb(s["right"])
        left_t = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).float().to(device)
        right_t = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).float().to(device)
        left_p, right_p, shape, _, _ = pad_batch(left_t, right_t)
        with torch.amp.autocast(enabled=True, device_type=device.type, dtype=torch.float16):
            pred, occ, conf = model(left_p, right_p)
        pred = image_crop(pred, shape).squeeze().float().cpu().numpy()

        gt = cv2.imread(str(s["disp"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        gt_depth = cv2.imread(str(s["depth"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        calib = json.loads((Path(servct_root) / s["experiment"] / "Rectified_calibration" / f"{s['frame']}.json").read_text())
        p1 = np.array(calib["P1"]["data"], dtype=np.float32).reshape(3, 4)
        p2 = np.array(calib["P2"]["data"], dtype=np.float32).reshape(3, 4)
        f = p1[0, 0]
        baseline_mm = abs(p2[0, 3] / p2[0, 0])
        pred_depth = f * baseline_mm / np.maximum(pred, 1e-6)
        mask = (gt > 0) & (gt_depth > 0) & np.isfinite(pred_depth)

        row = {"experiment": s["experiment"], "reference": s["reference"], "frame": s["frame"]}
        row.update(disp_metrics(pred, gt, mask))
        row.update(depth_metrics(pred_depth, gt_depth, mask))
        rows.append(row)

        max_disp = np.percentile(gt[mask], 99)
        err = np.abs(pred - gt)
        triptych = np.concatenate(
            [left, disp_vis(pred, max_disp)[..., ::-1], disp_vis(gt, max_disp)[..., ::-1], disp_vis(err, min(20, np.percentile(err[mask], 99)))[..., ::-1]],
            axis=1,
        )
        name = f"{s['experiment']}_{s['reference']}_{s['frame']}_left_pred_gt_err.png"
        cv2.imwrite(str(out_dir / name), triptych[..., ::-1])
        montage_rows.append((f"{s['experiment']}/{s['reference']}/{s['frame']}", triptych))

    keys = [
        "experiment", "reference", "frame", "valid_px",
        "mae_px", "rmse_px", "bad1_pct", "bad2_pct", "bad5_pct",
        "depth_mae_mm", "depth_rmse_mm", "depth_bad1mm_pct", "depth_bad2mm_pct", "depth_bad5mm_pct",
    ]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys[3:]}
    summary["frames"] = len(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    thumbs = []
    for label, triptych in montage_rows:
        thumb = cv2.resize(triptych, (1440, 288), interpolation=cv2.INTER_AREA)
        canvas = np.full((312, 1440, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[24:] = thumb
        thumbs.append(canvas)
    if thumbs:
        cv2.imwrite(str(out_dir / "montage_left_pred_gt_err.png"), np.concatenate(thumbs, axis=0)[..., ::-1])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--servct_root", default="../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT")
    parser.add_argument("--weights_dir", default="weights/pretrain_weights")
    parser.add_argument("--model_type", default="S", choices=["S", "M", "L", "XL"])
    parser.add_argument("--out_dir", default="output_servct_finetune_s2m2_S")
    parser.add_argument("--mode", default="honest_holdout", choices=["honest_holdout", "all_surgical"])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--crop_h", type=int, default=448)
    parser.add_argument("--crop_w", type=int, default=640)
    parser.add_argument("--refine_iter", type=int, default=3)
    parser.add_argument("--trainable", default="refiners", choices=["refiners", "all"])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    if args.mode == "honest_holdout":
        train_samples = ServCtDataset.collect(args.servct_root, ["Experiment_1"], ["Reference_CT"])
        val_samples = ServCtDataset.collect(args.servct_root, ["Experiment_2"], ["Reference_CT"])
    else:
        train_samples = ServCtDataset.collect(args.servct_root, ["Experiment_1", "Experiment_2"], ["Reference_CT", "Reference_RGB"])
        val_samples = ServCtDataset.collect(args.servct_root, ["Experiment_1", "Experiment_2"], ["Reference_CT"])

    train_ds = ServCtDataset(args.servct_root, train_samples, crop_size=(args.crop_h, args.crop_w), augment=True)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    model = build_model(args.model_type, args.refine_iter, use_positivity=True)
    load_pretrained(model, args.weights_dir, args.model_type)
    set_trainable(model, args.trainable)
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    logs = []
    loader_iter = iter(train_loader)
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        left = batch["left"].to(device)
        right = batch["right"].to(device)
        gt = batch["disp"].to(device)
        depth = batch["depth"].to(device)
        left_p, right_p, shape, gt_p, depth_p = pad_batch(left, right, gt, depth)
        mask = (gt_p > 0) & (depth_p > 0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=True, device_type=device.type, dtype=torch.float16):
            pred, occ, conf = model(left_p, right_p)
            l1 = F.smooth_l1_loss(pred[mask], gt_p[mask], beta=0.5)
            gl = gradient_loss(pred, gt_p, mask)
        conf_target = (pred.detach().float() - gt_p.float()).abs().lt(2.0).float()
        conf_loss = F.binary_cross_entropy(conf.float().clamp(1e-4, 1 - 1e-4)[mask], conf_target[mask])
        loss = l1 + 0.05 * gl + 0.05 * conf_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0 or step == 1:
            log = {"step": step, "loss": float(loss.detach().cpu()), "l1": float(l1.detach().cpu()), "grad": float(gl.detach().cpu()), "conf": float(conf_loss.detach().cpu())}
            logs.append(log)
            print(json.dumps(log))

    ckpt_path = out_dir / "s2m2_servct_finetuned.pth"
    torch.save({"state_dict": model.state_dict(), "config": vars(args)}, ckpt_path)
    (out_dir / "train_log.json").write_text(json.dumps(logs, indent=2))

    summary = evaluate(model, val_samples, args.servct_root, out_dir / "eval", device)
    print(json.dumps({"checkpoint": str(ckpt_path), "eval": summary}, indent=2))


if __name__ == "__main__":
    main()
