#!/usr/bin/env python3
"""ARGOS v2 one-seed ladder trainer.

Small and boring on purpose: samples causal clips, trains one model, evaluates
with the certified streaming evaluator functions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler

ROOT = Path("/dtu/p1/leopam/ARGOS")
sys.path.insert(0, str(ROOT))

from scripts.temporal_refinement.causal_bida.configs import CONFIGS, resolved_config  # noqa: E402
from scripts.temporal_refinement.eval_scripts.evaluate_argos_v2_streaming import (  # noqa: E402
    TARGETS,
    AUX_CACHE,
    aggregate_metrics,
    build_model,
    reliability_mask,
    stream_sequence,
    warp_with_support,
)

SPLIT = ROOT / "results/03_temporal_refinement/training/refiner_failure_analysis/proposed_balanced_split.json"
OUT_ROOT = ROOT / "results/03_temporal_refinement/argos_v2/one_seed_ladder"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


class ShardStore:
    def __init__(self, sequence_ids: list[str], max_frames: int = 0):
        self.sequence_ids = sequence_ids
        self.max_frames = max_frames
        self._cache: dict[str, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def load(self, seq: str, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        if seq not in self._cache:
            z = np.load(TARGETS / "targets" / f"{seq}.npz")
            aux = np.load(AUX_CACHE / f"{seq}.npz")
            n = z["raw_disp"].shape[0] if self.max_frames <= 0 else min(self.max_frames, z["raw_disp"].shape[0])
            self._cache[seq] = {
                "raw": torch.from_numpy(z["raw_disp"][:n].astype("float32"))[:, None],
                "gt": torch.from_numpy(z["gt_disp"][:n].astype("float32"))[:, None],
                "valid": torch.from_numpy((z["valid_mask"][:n] > 0).astype("float32"))[:, None],
                "rgb": torch.from_numpy(aux["rgb"][:n].astype("float32") / 255.0).permute(0, 3, 1, 2),
                "flow": torch.from_numpy(aux["warp_flow"][: max(n - 1, 0)].astype("float32")),
                "occ": torch.from_numpy(aux["occ"][: max(n - 1, 0)].astype("float32"))[:, None],
            }
        return {k: v.to(device, non_blocking=True) for k, v in self._cache[seq].items()}

    def sample_clip(self, rng: random.Random, clip_len: int, device: torch.device) -> tuple[str, dict[str, torch.Tensor]]:
        seq = rng.choice(self.sequence_ids)
        sh = self.load(seq, device)
        t = sh["raw"].shape[0]
        n = min(clip_len, t)
        start = rng.randint(0, max(0, t - n))
        sl = slice(start, start + n)
        fl = slice(start, start + max(n - 1, 0))
        return seq, {
            "raw": sh["raw"][sl],
            "gt": sh["gt"][sl],
            "valid": sh["valid"][sl],
            "rgb": sh["rgb"][sl],
            "flow": sh["flow"][fl],
            "occ": sh["occ"][fl],
        }


def load_splits(max_sequences: int = 0, max_frames: int = 0):
    split = json.loads(SPLIT.read_text())
    out = {}
    for name, seqs in split.items():
        seqs = list(seqs)
        if max_sequences > 0:
            seqs = seqs[:max_sequences]
        out[name] = ShardStore(seqs, max_frames=max_frames)
    return out


def losses(raw, refined, gt, valid, flow, occ, cfg):
    mask = valid > 0.5
    zero = refined.sum() * 0
    spatial = (refined - gt).abs()[mask].mean() if mask.any() else zero
    tgm_terms = []
    for i in range(1, refined.shape[0]):
        m = (valid[i] > 0.5) & (valid[i - 1] > 0.5)
        if m.any():
            tgm_terms.append(((refined[i] - refined[i - 1]) - (gt[i] - gt[i - 1])).abs()[m].mean())
    tgm = torch.stack(tgm_terms).mean() if tgm_terms else zero
    warp_terms = []
    for i in range(1, refined.shape[0]):
        warped, inb = warp_with_support(refined[i - 1 : i], flow[i - 1 : i])
        rel = reliability_mask(valid[i : i + 1], valid[i - 1 : i], flow[i - 1 : i], occ[i - 1 : i])
        support = (rel > 0.5) & (inb > 0.5)
        if support.any():
            warp_terms.append((refined[i : i + 1] - warped).abs()[support].mean())
    warp = torch.stack(warp_terms).mean() if warp_terms else zero
    applied = refined - raw
    good = mask & ((raw - gt).abs() < 1)
    safe = applied.abs()[good].mean() if good.any() else zero
    sparse = applied.abs()[mask].mean() if mask.any() else zero
    total = cfg["spatial_weight"] * spatial + cfg["tgm_weight"] * tgm + cfg["warp_weight"] * warp
    if cfg.get("safe_losses"):
        total = total + cfg["safe_weight"] * safe + cfg["sparse_weight"] * sparse
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_spatial": float(spatial.detach().cpu()),
        "loss_tgm": float(tgm.detach().cpu()),
        "loss_warp": float(warp.detach().cpu()),
        "loss_safe": float(safe.detach().cpu()),
        "loss_sparse": float(sparse.detach().cpu()),
        "valid_pixels": int(mask.sum().detach().cpu()),
        "valid_temporal_pairs": int(sum(((valid[i] > 0.5) & (valid[i - 1] > 0.5)).sum().item() for i in range(1, valid.shape[0]))),
    }


def eval_store(model, store: ShardStore, cfg, device, split_name: str, max_sequences: int = 0):
    model.eval()
    frame_rows, seq_rows = [], []
    totals = []
    with torch.no_grad():
        seqs = store.sequence_ids[: max_sequences or None]
        for seq in seqs:
            sh = store.load(seq, device)
            res = stream_sequence(model, sh["raw"], sh["valid"], sh["rgb"], sh["flow"], sh["occ"], mode=cfg["mode"])
            metrics = aggregate_metrics(sh["raw"], res.refined, sh["gt"], sh["valid"])
            seq_rows.append({"split": split_name, "sequence_id": seq, **metrics, "temporal_pairs": res.temporal_pair_count})
            totals.append(metrics)
            err_raw = (sh["raw"] - sh["gt"]).abs()
            err_ref = (res.refined - sh["gt"]).abs()
            for i in range(sh["raw"].shape[0]):
                m = sh["valid"][i] > 0.5
                frame_rows.append({
                    "split": split_name,
                    "sequence_id": seq,
                    "frame_index": i,
                    "raw_mae": float(err_raw[i][m].mean()) if m.any() else float("nan"),
                    "refined_mae": float(err_ref[i][m].mean()) if m.any() else float("nan"),
                    "modified_pixel_ratio": float(((res.refined[i] - sh["raw"][i]).abs()[m] > 0.1).float().mean()) if m.any() else float("nan"),
                })
    agg = {}
    for k in ["raw_mae", "refined_mae", "delta_mae", "raw_bad3", "refined_bad3", "new_bad3", "modified_pixel_ratio"]:
        vals = [r[k] for r in totals if k in r and np.isfinite(r[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    agg["split"] = split_name
    agg["sequences"] = len(totals)
    return agg, seq_rows, frame_rows


def save_checkpoint(path, model, opt, scaler, cfg, step, best_metric):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": None if opt is None else opt.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "cfg": cfg,
        "step": step,
        "best_metric": best_metric,
    }, path)


def run(args):
    cfg = resolved_config(args.config, steps=args.steps, clip_len=args.clip_len, eval_every=args.eval_every)
    if args.smoke:
        cfg.update({"steps": args.smoke_steps, "eval_every": args.smoke_steps})
    out = args.output_root / f"{args.config}_seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "environment_summary.txt").write_text(
        f"git_commit={os.popen('git rev-parse HEAD').read().strip()}\npython={sys.version}\n"
    )
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    splits = load_splits(args.max_sequences, args.max_frames)
    model = build_model(cfg["model"]).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = None if cfg.get("eval_only") else torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = GradScaler(enabled=(device.type == "cuda" and args.amp))
    start_step = 0
    best_metric = float("inf")
    latest = out / "checkpoints/latest.pt"
    if args.resume and latest.exists():
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        if opt is not None and ckpt.get("optimizer"):
            opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt["step"])
        best_metric = float(ckpt["best_metric"])
    if cfg.get("eval_only"):
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    log_rows = []
    t0 = time.time()
    for step in range(start_step + 1, cfg["steps"] + 1):
        if cfg.get("eval_only"):
            break
        model.train()
        _seq, batch = splits["train"].sample_clip(rng, cfg["clip_len"], device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and args.amp)):
            res = stream_sequence(model, batch["raw"], batch["valid"], batch["rgb"], batch["flow"], batch["occ"], mode=cfg["mode"], detach_state=False)
            loss, parts = losses(batch["raw"], res.refined, batch["gt"], batch["valid"], batch["flow"], batch["occ"], cfg)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step == 1 or step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            val_agg, _, _ = eval_store(model, splits["val"], cfg, device, "val", max_sequences=args.eval_sequences)
            row = {"step": step, **parts, **{f"val_{k}": v for k, v in val_agg.items() if k != "split"}}
            log_rows.append(row)
            write_csv(out / "train_log.csv", log_rows)
            metric = float(val_agg["refined_mae"])
            if metric < best_metric:
                best_metric = metric
                save_checkpoint(out / "checkpoints/best.pt", model, opt, scaler, cfg, step, best_metric)
            save_checkpoint(latest, model, opt, scaler, cfg, step, best_metric)
    # final eval on val only for ladder gate; no test/OOD here.
    if not cfg.get("eval_only") and (out / "checkpoints/best.pt").exists():
        model.load_state_dict(torch.load(out / "checkpoints/best.pt", map_location=device)["model"])
    val_agg, seq_rows, frame_rows = eval_store(model, splits["val"], cfg, device, "val", max_sequences=args.eval_sequences)
    write_csv(out / "per_sequence_metrics.csv", seq_rows)
    write_csv(out / "per_frame_metrics.csv", frame_rows)
    (out / "aggregate_metrics.json").write_text(json.dumps({
        "config": args.config,
        "params": params,
        "elapsed_s": time.time() - t0,
        "val": val_agg,
        "best_metric": best_metric,
    }, indent=2) + "\n")
    print(json.dumps({"config": args.config, "params": params, "val": val_agg, "out": str(out)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--output-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--clip-len", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-sequences", type=int, default=0)
    ap.add_argument("--max-sequences", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--checkpoint", type=Path)
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
