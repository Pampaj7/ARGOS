import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
S2M2_SRC = REPO_ROOT / "stereo/s2m2/src"
sys.path.insert(0, str(S2M2_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_scared_s2m2 import MODEL_CONFIG, build_model, infer, load_pretrained, read_rgb
from s2m2.core.utils.image_utils import image_pad


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_trainable(model, mode):
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = False
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


class WarpedDataset(Dataset):
    def __init__(self, metadata_csv, crop_size=(512, 768), max_width=1024, min_valid_ratio=0.2, augment=True):
        rows = list(csv.DictReader(open(metadata_csv)))
        self.samples = [
            r
            for r in rows
            if r["left_path"]
            and r["notes"] == ""
            and float(r["valid_pixel_ratio"]) >= min_valid_ratio
            and Path(r["left_path"]).exists()
            and Path(r["disparity_float32_path"]).exists()
        ]
        self.crop_size = crop_size
        self.max_width = max_width
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _resize(self, left, right, disp, mask):
        h, w = disp.shape
        if not self.max_width or w <= self.max_width:
            return left, right, disp, mask
        scale = self.max_width / float(w)
        new_size = (self.max_width, int(round(h * scale)))
        left = cv2.resize(left, new_size, interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, new_size, interpolation=cv2.INTER_LINEAR)
        disp = cv2.resize(disp, new_size, interpolation=cv2.INTER_LINEAR) * scale
        mask = cv2.resize(mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        return left, right, disp, mask

    def _crop(self, left, right, disp, mask):
        crop_h, crop_w = self.crop_size
        h, w = disp.shape
        if h <= crop_h or w <= crop_w:
            return left, right, disp, mask
        valid = mask.astype(np.uint8)
        for _ in range(20):
            y = random.randint(0, h - crop_h)
            x = random.randint(0, w - crop_w)
            if valid[y : y + crop_h, x : x + crop_w].mean() > 0.15:
                break
        return (
            left[y : y + crop_h, x : x + crop_w],
            right[y : y + crop_h, x : x + crop_w],
            disp[y : y + crop_h, x : x + crop_w],
            mask[y : y + crop_h, x : x + crop_w],
        )

    def _augment(self, left, right):
        if not self.augment:
            return left, right
        gain = random.uniform(0.9, 1.1)
        bias = random.uniform(-6, 6)
        left = np.clip(left.astype(np.float32) * gain + bias, 0, 255)
        right = np.clip(right.astype(np.float32) * gain + bias, 0, 255)
        return left.astype(np.uint8), right.astype(np.uint8)

    def __getitem__(self, idx):
        s = self.samples[idx]
        left = read_rgb(s["left_path"])
        right = read_rgb(s["right_path"])
        disp = np.load(s["disparity_float32_path"]).astype(np.float32)
        mask = np.load(s["valid_mask_path"]).astype(bool)
        left, right, disp, mask = self._resize(left, right, disp, mask)
        left, right, disp, mask = self._crop(left, right, disp, mask)
        left, right = self._augment(left, right)
        return {
            "left": torch.from_numpy(left.copy()).permute(2, 0, 1).float(),
            "right": torch.from_numpy(right.copy()).permute(2, 0, 1).float(),
            "disp": torch.from_numpy(disp.copy()).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask.copy()).unsqueeze(0).bool(),
        }


def pad_batch(left, right, disp, mask):
    h, w = left.shape[-2:]
    left_p = image_pad(left, 32)
    right_p = image_pad(right, 32)
    pad_h = left_p.shape[-2] - h
    pad_w = left_p.shape[-1] - w
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return left_p, right_p, F.pad(disp, pad, mode="replicate"), F.pad(mask.float(), pad, mode="constant", value=0).bool()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", default="results/scared_warped_train_subset_metadata.csv")
    parser.add_argument("--weights_dir", default="stereo/s2m2/weights/pretrain_weights")
    parser.add_argument("--out_dir", default="results/scared_s2m2_xl_warped_finetune_run1")
    parser.add_argument("--model_type", default="XL", choices=["S", "M", "L", "XL"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--max_width", type=int, default=1024)
    parser.add_argument("--crop_h", type=int, default=512)
    parser.add_argument("--crop_w", type=int, default=768)
    parser.add_argument("--min_valid_ratio", type=float, default=0.2)
    parser.add_argument("--trainable", default="refiners", choices=["refiners", "all"])
    parser.add_argument("--refine_iter", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    ds = WarpedDataset(
        args.metadata_csv,
        crop_size=(args.crop_h, args.crop_w),
        max_width=args.max_width,
        min_valid_ratio=args.min_valid_ratio,
        augment=True,
    )
    if len(ds) == 0:
        raise RuntimeError("No training samples after filtering")
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, drop_last=True)

    model = build_model(args.model_type, args.refine_iter, device)
    load_pretrained(model, args.weights_dir, args.model_type)
    set_trainable(model, args.trainable)
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    logs = []
    it = iter(loader)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        left = batch["left"].to(device)
        right = batch["right"].to(device)
        gt = batch["disp"].to(device)
        mask = batch["mask"].to(device)
        left_p, right_p, gt_p, mask_p = pad_batch(left, right, gt, mask)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred, _occ, _conf = model(left_p, right_p)
            loss = F.smooth_l1_loss(pred[mask_p], gt_p[mask_p], beta=0.5)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 20 == 0:
            log = {"step": step, "loss": float(loss.detach().cpu()), "elapsed_s": time.perf_counter() - start}
            logs.append(log)
            print(json.dumps(log), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            ckpt = ckpt_dir / f"step_{step:06d}.pth"
            torch.save({"state_dict": model.state_dict(), "config": vars(args), "step": step}, ckpt)

    (out_dir / "train_log.json").write_text(json.dumps(logs, indent=2))
    print(json.dumps({"samples": len(ds), "out_dir": str(out_dir), "last_checkpoint": str(ckpt)}, indent=2))


if __name__ == "__main__":
    main()
