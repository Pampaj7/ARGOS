#!/usr/bin/env python3
"""Train the Candidate-Fusion Refiner (fable wildcard experiment).

Mixed batches: 50% full-GT crops (no candidate maps -> raw-only degraded mode) and 50%
selected-clip crops carrying the full candidate stack (raw, fixed-EMA, adaptive,
RAFT-small-warped, SAV) with random candidate dropout for robustness. One uniform
GT-supervised loss for all batches. Same supervision protocol as v3.2b/c (which also
trained on these selected clips via oracle targets), so selected-clip comparisons
against those baselines are like-for-like. No new teacher inference — all candidate
maps already exist in the target npz files.

Single GPU, no DataParallel, cudnn.benchmark off (EGBM crash lessons).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

from train_tiny_refiner_v1_full_gt import (  # noqa: E402
    DEFAULT_TARGETS_ROOT,
    charbonnier,
    finite_mean,
    load_shards,
    masked_mean,
    parse_bool,
    write_csv,
)
from train_tiny_refiner_v3_1_staged_abstention import (  # noqa: E402
    DEFAULT_BALANCED_SPLIT,
    BalancedCropDataset,
    FullFrameDataset,
    load_samples_with_split,
    make_features_from_raws,
    write_csv_union,
)
from train_tiny_refiner_v3_2_hybrid_oracle import make_loader  # noqa: E402
from train_experimental_refiner_vx import aggregate_frames, frame_metrics_egbm, full_gt_eval, score_epoch  # noqa: E402
from calibrate_v3_3_failure_mode_thresholds import PATHOLOGICAL_MODES  # noqa: E402
from evaluate_v3_1_on_selected_oracle_clips import read_csv  # noqa: E402
from candidate_fusion_refiner import CANDIDATE_KEYS, N_CANDIDATES, cfr_medium  # noqa: E402


DEFAULT_ORACLE_TARGETS = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/fable_wildcard_experiment")
BASELINES = {
    "v3.2c": {"selected_mae": 11.0054, "gap_pct": 7.03, "patho_new_bad3": 15.77, "clean_new_bad3": 0.89, "full_gt_test_mae": 4.6145},
    "v4_tiny": {"selected_mae": 11.0669, "gap_pct": 5.67, "patho_new_bad3": 0.33, "full_gt_test_mae": 4.7763},
    "SOG": {"selected_mae": 11.0909, "gap_pct": 5.14, "patho_new_bad3": 5.77, "full_gt_test_mae": 4.6221},
    "v3.3_threshold_only": {"selected_mae": 11.1062, "gap_pct": 4.80, "patho_new_bad3": 6.69},
    "v3.3b": {"selected_mae": 11.0059, "gap_pct": 7.02, "patho_new_bad3": 15.25},
    "EGBM": {"note": "still training at time of this run"},
}


@torch.no_grad()
def benchmark_cfr(model: nn.Module, args: argparse.Namespace, device: torch.device, batch_size: int = 32) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    x = torch.randn(batch_size, args.context_frames * 2 + 8, 256, 320, device=device)
    raw = x[:, 0:1] * 64.0
    candidates = raw.expand(-1, N_CANDIDATES, -1, -1).contiguous()
    flags = torch.zeros(batch_size, N_CANDIDATES, device=device)
    flags[:, 0] = 1.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    for _ in range(10):
        model(x, args.residual_scale, candidates, flags)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        model(x, args.residual_scale, candidates, flags)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_per_frame = 1000.0 * (time.perf_counter() - t0) / (50 * batch_size)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    _gl, gate, residual_out, diag = model(x[:2], args.residual_scale, candidates[:2], flags[:2])
    summary = {
        "model": "candidate_fusion_refiner_medium",
        "params": sum(p.numel() for p in model.parameters()),
        "batch_size": batch_size,
        "resolution": "256x320",
        "fp32_ms_per_frame_batched": round(ms_per_frame, 4),
        "peak_vram_mb": round(float(peak_mb), 1),
        "estimated_system_total_ms_with_s2m2_62ms": round(62.0 + ms_per_frame, 4),
        "identity_at_init_max_abs_delta_px": float(residual_out.abs().max().detach().cpu()),
        "gate_mean_at_init": float(gate.mean().detach().cpu()),
        "raw_weight_mean_at_init": float(diag["weights"][:, 0:1].mean().detach().cpu()),
        "runtime_budget_met_35ms": bool(ms_per_frame < 35.0),
    }
    rows = [{
        "model": summary["model"],
        "params": summary["params"],
        "batch_size": batch_size,
        "resolution": summary["resolution"],
        "precision": "fp32",
        "ms_per_frame": summary["fp32_ms_per_frame_batched"],
        "peak_vram_mb": summary["peak_vram_mb"],
        "identity_at_init_max_abs_delta_px": round(summary["identity_at_init_max_abs_delta_px"], 8),
    }]
    return summary, rows


class CandidateClip:
    def __init__(self, clip_dir: Path, clip_index: dict[str, dict[str, str]]):
        rows = read_csv(clip_dir / "frame_target_index.csv")
        meta = json.loads((clip_dir / "clip_metadata.json").read_text())
        self.clip_id = clip_dir.name
        self.sequence_id = meta["sequence_id"]
        self.failure_mode = clip_index.get(clip_dir.name, {}).get("dominant_failure_mode", "")
        self.frame_ids = [r["frame_id"] for r in rows]
        frames = [np.load(r["target_path"]) for r in rows]
        self.gts = np.stack([f["gt_disp"].astype(np.float32) for f in frames])
        self.valids = np.stack([f["valid_mask"].astype(np.float32) for f in frames])
        self.oracle = np.stack([f["oracle_all_available_disp"].astype(np.float32) for f in frames])
        cands, flags = [], []
        for key in CANDIDATE_KEYS:
            if key in frames[0].files:
                cands.append(np.stack([f[key].astype(np.float32) for f in frames]))
                flags.append(1.0)
            else:
                cands.append(np.stack([f["raw_disp"].astype(np.float32) for f in frames]))
                flags.append(0.0)
        self.candidates = np.stack(cands, axis=1)  # (T, K, H, W)
        self.flags = np.array(flags, dtype=np.float32)
        self.raws = self.candidates[:, 0]


def load_candidate_clips(root: Path) -> list[CandidateClip]:
    clip_index = {r["clip_id"]: r for r in read_csv(root / "clip_targets_index.csv")}
    return [CandidateClip(d, clip_index) for d in sorted((root / "clips").iterdir()) if d.is_dir()]


class CandidateCropDataset(Dataset):
    """Selected-clip crops with the full candidate stack; biased toward high-error regions."""

    def __init__(self, clips: list[CandidateClip], args: argparse.Namespace, crops_per_epoch: int):
        self.clips = clips
        self.args = args
        self.crops_per_epoch = crops_per_epoch
        self.frame_pool = [(ci, fi) for ci, c in enumerate(clips) for fi in range(len(c.frame_ids))]
        self.rng = random.Random(2468)

    def __len__(self) -> int:
        return self.crops_per_epoch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        a = self.args
        ci, fi = self.frame_pool[self.rng.randrange(len(self.frame_pool))]
        clip = self.clips[ci]
        h, w = clip.raws.shape[1:]
        s = min(a.crop_size, h, w)
        err = np.abs(clip.raws[fi] - clip.gts[fi]) * (clip.valids[fi] > 0)
        best = (-1.0, 0, 0)
        for _ in range(a.crop_candidate_tries):
            y = self.rng.randint(0, max(0, h - s))
            x = self.rng.randint(0, max(0, w - s))
            score = float((err[y : y + s, x : x + s] >= a.bad_threshold_px).mean()) if self.rng.random() < 0.7 else self.rng.random()
            if score >= best[0]:
                best = (score, y, x)
        _, y, x = best
        ys, xs = slice(y, y + s), slice(x, x + s)
        ids = [max(0, fi - i) for i in range(a.context_frames)]
        raws = clip.raws[ids, ys, xs]
        valids = clip.valids[ids, ys, xs]
        xfeat, _e, _v = make_features_from_raws(raws, valids)
        flags = clip.flags.copy()
        cands = clip.candidates[fi, :, ys, xs].copy()
        # candidate dropout: teach robustness to missing candidates (raw always kept)
        for k in range(1, N_CANDIDATES):
            if flags[k] > 0 and self.rng.random() < a.candidate_dropout:
                cands[k] = cands[0]
                flags[k] = 0.0
        return {
            "x": torch.from_numpy(xfeat),
            "raw": torch.from_numpy(raws[0][None]),
            "gt": torch.from_numpy(clip.gts[fi, ys, xs][None]),
            "valid": torch.from_numpy(valids[0][None]),
            "candidates": torch.from_numpy(cands),
            "flags": torch.from_numpy(flags),
        }


def cfr_loss_batch(model: nn.Module, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device, source: str) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    raw = batch["raw"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    candidates = batch.get("candidates")
    flags = batch.get("flags")
    if candidates is not None:
        candidates = candidates.to(device, non_blocking=True)
        flags = flags.to(device, non_blocking=True)
    _gl, gate, residual_out, diag = model(x, args.residual_scale, candidates, flags)
    final = raw + residual_out
    raw_err = torch.abs(raw - gt)
    final_err = torch.abs(final - gt)
    zero = residual_out.sum() * 0.0
    acc = masked_mean(torch.clamp(charbonnier(final - gt), max=args.robust_loss_clip_px), valid)
    below3 = valid * (raw_err < args.bad_threshold_px).float()
    nb3 = masked_mean(torch.relu(final_err - args.bad_threshold_px), below3) if float(below3.sum()) > 0 else zero
    raw_good = valid * (raw_err < args.good_threshold_px).float()
    preserve = masked_mean(torch.abs(final - raw), raw_good) if float(raw_good.sum()) > 0 else zero
    loss = args.acc_weight * acc + args.new_bad3_weight * nb3 + args.preserve_weight * preserve
    return loss, {
        f"{source}_loss": float(loss.detach().cpu()),
        f"{source}_acc": float(acc.detach().cpu()),
        f"{source}_nb3": float(nb3.detach().cpu()),
        f"{source}_preserve": float(preserve.detach().cpu()),
        f"{source}_gate_mean": float(masked_mean(gate, valid).detach().cpu()),
        f"{source}_raw_weight_mean": float(masked_mean(diag["weights"][:, 0:1], valid).detach().cpu()),
    }


def train_one_epoch(model: nn.Module, loaders: dict[str, DataLoader], optimizer: torch.optim.Optimizer, args: argparse.Namespace, device: torch.device, rng: random.Random) -> dict[str, float]:
    model.train()
    order = [name for name, loader in loaders.items() for _ in range(len(loader))]
    rng.shuffle(order)
    iters = {name: iter(loader) for name, loader in loaders.items()}
    rows: list[dict[str, float]] = []
    for source in order:
        try:
            batch = next(iters[source])
        except StopIteration:
            continue
        loss, metrics = cfr_loss_batch(model, batch, args, device, source)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        rows.append(metrics)
    keys = {k for r in rows for k in r}
    return {k: finite_mean([r[k] for r in rows if k in r]) for k in sorted(keys)}


@torch.no_grad()
def predict_clip_cfr(model: nn.Module, clip: CandidateClip, args: argparse.Namespace, device: torch.device, use_candidates: bool = True) -> np.ndarray:
    refined_all = []
    n = len(clip.frame_ids)
    for s_ in range(0, n, args.eval_clip_batch):
        e_ = min(n, s_ + args.eval_clip_batch)
        xs = []
        for i in range(s_, e_):
            ids = [max(0, i - k) for k in range(args.context_frames)]
            xf, _e2, _v2 = make_features_from_raws(clip.raws[ids], clip.valids[ids])
            xs.append(xf)
        xb = torch.from_numpy(np.stack(xs)).to(device)
        if use_candidates:
            cands = torch.from_numpy(clip.candidates[s_:e_]).to(device)
            flags = torch.from_numpy(np.tile(clip.flags, (e_ - s_, 1))).to(device)
        else:
            cands = flags = None
        _gl, _g, residual_out, _diag = model(xb, args.residual_scale, cands, flags)
        refined_all.append((torch.from_numpy(clip.raws[s_:e_]).to(device) + residual_out[:, 0]).cpu().numpy())
    return np.concatenate(refined_all)


def eval_selected_cfr(model: nn.Module, clips: list[CandidateClip], args: argparse.Namespace, device: torch.device, use_candidates: bool = True) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    model.eval()
    preds = {c.clip_id: predict_clip_cfr(model, c, args, device, use_candidates) for c in clips}
    out: dict[str, dict[str, float]] = {}
    for name, group in (("all", clips), ("patho", [c for c in clips if c.failure_mode in PATHOLOGICAL_MODES]), ("clean", [c for c in clips if c.failure_mode not in PATHOLOGICAL_MODES])):
        frames = [f for c in group for f in frame_metrics_egbm(c, preds[c.clip_id])]
        out[name] = aggregate_frames(frames)
    return out, preds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--oracle-targets-root", type=Path, default=DEFAULT_ORACLE_TARGETS)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--balanced-split-json", type=Path, default=DEFAULT_BALANCED_SPLIT)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--crop-candidate-tries", type=int, default=12)
    p.add_argument("--crops-per-epoch", type=int, default=60000)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=48)
    p.add_argument("--eval-clip-batch", type=int, default=16)
    p.add_argument("--full-gt-batch-ratio", type=float, default=0.50)
    p.add_argument("--candidate-batch-ratio", type=float, default=0.50)
    p.add_argument("--candidate-dropout", type=float, default=0.20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--residual-scale", type=float, default=3.0)
    p.add_argument("--bad-threshold-px", type=float, default=3.0)
    p.add_argument("--good-threshold-px", type=float, default=1.0)
    p.add_argument("--acc-weight", type=float, default=1.0)
    p.add_argument("--new-bad3-weight", type=float, default=2.0)
    p.add_argument("--preserve-weight", type=float, default=0.5)
    p.add_argument("--robust-loss-clip-px", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--max-auc-pixels", type=int, default=200000)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--oracle-min-improvement-px", type=float, default=1.0)
    p.add_argument("--oracle-hard-only", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--fresh", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    args = p.parse_args()
    total = args.full_gt_batch_ratio + args.candidate_batch_ratio
    args.full_gt_batch_ratio /= total
    args.candidate_batch_ratio /= total
    return args


def main() -> int:
    args = parse_args()
    if (args.fresh or args.overwrite) and args.output_root.exists():
        shutil.rmtree(args.output_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"{args.output_root} exists; pass --overwrite true")
    (args.output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.output_root / "diagnostics").mkdir(exist_ok=True)
    sys.stdout = (args.output_root / "stdout.log").open("w", buffering=1)
    sys.stderr = (args.output_root / "stderr.log").open("w", buffering=1)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    start = time.perf_counter()

    model = cfr_medium(16, args.residual_scale).to(device)
    params = sum(p.numel() for p in model.parameters())
    bench_summary, bench_rows = benchmark_cfr(model, args, device)

    splits, by_split = load_samples_with_split(args.targets_root, args.balanced_split_json, args.max_frames)
    shards = load_shards(by_split["train"] + by_split["val"] + by_split["test"])
    clips = load_candidate_clips(args.oracle_targets_root)
    full_ds = {split: FullFrameDataset(samples, shards, args.context_frames) for split, samples in by_split.items()}
    eval_loaders = {split: make_loader(ds, args.eval_batch_size, max(0, args.num_workers // 2), False, args.prefetch_factor) for split, ds in full_ds.items()}

    gt_crops = int(round(args.crops_per_epoch * args.full_gt_batch_ratio))
    cand_crops = args.crops_per_epoch - gt_crops
    gt_args = argparse.Namespace(**{**vars(args), "crops_per_epoch": gt_crops})
    loaders = {
        "gt": make_loader(BalancedCropDataset(by_split["train"], shards, gt_args), args.batch_size, args.num_workers, True, args.prefetch_factor),
        "cand": make_loader(CandidateCropDataset(clips, args, cand_crops), args.batch_size, max(2, args.num_workers // 3), True, args.prefetch_factor),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_lines = [
        f"device={device} params={params} model=cfr_medium",
        f"benchmark fp32_ms={bench_summary['fp32_ms_per_frame_batched']} peak_mb={bench_summary['peak_vram_mb']} identity_delta={bench_summary['identity_at_init_max_abs_delta_px']}",
        f"clips={len(clips)} candidate_flags={[list(c.flags) for c in clips][:1]}",
        f"frames={{'train': {len(full_ds['train'])}, 'val': {len(full_ds['val'])}, 'test': {len(full_ds['test'])}}}",
        f"crops gt={gt_crops} cand={cand_crops} batch={args.batch_size} epochs={args.epochs} cand_dropout={args.candidate_dropout}",
        f"weights acc={args.acc_weight} nb3={args.new_bad3_weight} preserve={args.preserve_weight}",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")
    (args.output_root / "benchmark_summary.json").write_text(json.dumps(bench_summary, indent=2, default=str) + "\n")
    with (args.output_root / "benchmark_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bench_rows[0]))
        writer.writeheader()
        writer.writerows(bench_rows)

    def log(line: str) -> None:
        run_lines.append(line)
        (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    rng = random.Random(555)
    train_rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(model, loaders, optimizer, args, device, rng)
        sel, _ = eval_selected_cfr(model, clips, args, device, use_candidates=True)
        fg = full_gt_eval(model, eval_loaders["val"], device, args.bad_threshold_px)
        fg_row = {"raw_mae": fg["raw_mae"], "refined_mae": fg["refined_mae"]}
        score = score_epoch(sel, fg_row)
        train_rows.append({
            "epoch": epoch, "score": score, **metrics,
            "sel_all_mae": sel["all"]["refined_mae"], "sel_all_gap": sel["all"]["oracle_gap_recovered_pct"],
            "sel_patho_new_bad3": sel["patho"]["new_bad3_frame_mean_pct"], "sel_clean_new_bad3": sel["clean"]["new_bad3_frame_mean_pct"],
            "sel_modified": sel["all"]["modified_pct"], "fullgt_val_raw": fg["raw_mae"], "fullgt_val_refined": fg["refined_mae"],
        })
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(), "args": vars(args), "splits": splits,
                "parameter_count": params, "epoch": epoch, "selected_metrics": sel, "full_gt_val_metrics": fg,
            }, args.output_root / "checkpoints" / "best.pt")
        log(
            f"epoch={epoch} score={score:.4f} sel_mae={sel['all']['refined_mae']:.4f} gap={sel['all']['oracle_gap_recovered_pct']:.2f}% "
            f"patho_nb3={sel['patho']['new_bad3_frame_mean_pct']:.2f}% clean_nb3={sel['clean']['new_bad3_frame_mean_pct']:.2f}% "
            f"mod={sel['all']['modified_pct']:.2f}% raw_w={metrics.get('cand_raw_weight_mean', float('nan')):.3f} "
            f"fullgt_val={fg['raw_mae']:.4f}->{fg['refined_mae']:.4f}"
        )
        if epoch - best_epoch >= args.early_stop_patience:
            log(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            break

    best = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])

    sel, preds = eval_selected_cfr(model, clips, args, device, use_candidates=True)
    sel_nocand, _ = eval_selected_cfr(model, clips, args, device, use_candidates=False)
    frame_rows = []
    for c in clips:
        for i, f in enumerate(frame_metrics_egbm(c, preds[c.clip_id])):
            frame_rows.append({"clip_id": c.clip_id, "sequence_id": c.sequence_id, "frame_id": c.frame_ids[i], "dominant_failure_mode": c.failure_mode, **f})
    fg_final = {split: full_gt_eval(model, eval_loaders[split], device, args.bad_threshold_px) for split in ("val", "test")}

    write_csv_union(args.output_root / "train_log.csv", train_rows)
    write_csv_union(args.output_root / "selected_oracle_metrics.csv", frame_rows)
    write_csv(args.output_root / "full_gt_val_metrics.csv", [fg_final["val"]])
    write_csv(args.output_root / "full_gt_test_metrics.csv", [fg_final["test"]])
    write_csv(args.output_root / "pathological_metrics.csv", [sel["patho"]])
    write_csv(args.output_root / "clean_metrics.csv", [sel["clean"]])

    success = {
        "strong_selected_mae_below_v32c": bool(sel["all"]["refined_mae"] < 11.0054),
        "strong_gap_above_v32c": bool(sel["all"]["oracle_gap_recovered_pct"] > 7.03),
        "strong_patho_new_bad3_below_8pct": bool(sel["patho"]["new_bad3_frame_mean_pct"] < 8.0),
        "strong_clean_new_bad3_at_most_1pct": bool(sel["clean"]["new_bad3_frame_mean_pct"] <= 1.0),
        "strong_full_gt_test_beats_raw": bool(fg_final["test"]["refined_mae"] < fg_final["test"]["raw_mae"]),
        "excellent_selected_mae_at_most_10_90": bool(sel["all"]["refined_mae"] <= 10.90),
        "excellent_gap_at_least_9pct": bool(sel["all"]["oracle_gap_recovered_pct"] >= 9.0),
    }
    summary = {
        "model": "candidate_fusion_refiner_medium",
        "output_root": str(args.output_root),
        "params": params,
        "best_epoch": best["epoch"],
        "epochs_run": epoch,
        "elapsed_seconds": time.perf_counter() - start,
        "benchmark": bench_summary,
        "selected_all_with_candidates": sel["all"],
        "selected_pathological": sel["patho"],
        "selected_clean": sel["clean"],
        "selected_all_no_candidates_degraded_mode": sel_nocand["all"],
        "full_gt_val": fg_final["val"],
        "full_gt_test": fg_final["test"],
        "baselines": BASELINES,
        "success_criteria": success,
        "no_teacher_inference": True,
        "candidate_maps_used": list(CANDIDATE_KEYS),
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (args.output_root / "README.md").write_text(f"""# Candidate-Fusion Refiner Wildcard

Chosen experiment: medium Candidate-Fusion Refiner (CFR), a 3-scale encoder-decoder that
mixes the existing candidate disparity maps (`raw`, fixed EMA, adaptive no-RAFT,
RAFT-small, SAV) with a small gated residual. The point is direct: the selected-clip
oracle is a per-pixel argmin over these candidates, so CFR tests whether learning that
selection beats another raw-S2M2-only controller.

Difference from v3.2c/v4/SOG/EGBM: those models infer corrections from S2M2-derived
features. CFR receives the same candidate maps the oracle uses when they exist, and
falls back to raw-only mode on full-GT shards.

Hypothesis: candidate fusion can recover more SAV/oracle headroom while preserving
full-GT safety through mixed full-GT/candidate training and candidate dropout.

Risks: selected clips are tiny, SAV/RAFT candidates are offline teachers, and the model
may overfit candidate availability instead of learning a deployable cheap-candidate
policy. Success means selected MAE below 11.0054, oracle gap above 7.03%, pathological
new-Bad3 below 8%, clean new-Bad3 <=1%, full-GT test beats raw, and runtime below
35 ms/frame.

## Result

- Params: `{params:,}`
- Runtime: `{bench_summary['fp32_ms_per_frame_batched']}` ms/frame fp32 batched
- Peak VRAM: `{bench_summary['peak_vram_mb']}` MB
- Init identity delta: `{bench_summary['identity_at_init_max_abs_delta_px']:.6f}` px
- Best epoch: `{best['epoch']}`
- Selected MAE: `{sel['all']['refined_mae']:.4f}`
- Oracle gap recovered: `{sel['all']['oracle_gap_recovered_pct']:.2f}%`
- Pathological new-Bad3: `{sel['patho']['new_bad3_frame_mean_pct']:.2f}%`
- Clean new-Bad3: `{sel['clean']['new_bad3_frame_mean_pct']:.2f}%`
- Full-GT test: raw `{fg_final['test']['raw_mae']:.4f}` -> refined `{fg_final['test']['refined_mae']:.4f}`

Success criteria: `{json.dumps(success)}`
""")
    print(json.dumps(success, indent=2))
    print(json.dumps({"with_candidates": sel["all"], "no_candidates": sel_nocand["all"], "patho": sel["patho"], "clean": sel["clean"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
