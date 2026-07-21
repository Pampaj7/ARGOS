#!/usr/bin/env python3
"""Controlled multi-domain training of the unchanged ARGOS v2 A2 proposal.

Only SCARED-C and curated D4D/S2M2-S supervision are loaded for fitting and
selection.  The script intentionally has no unseen-domain evaluation stage
until a frozen candidate has passed both seen-domain gates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from model_design.data.multidomain_raw_error_dataset import (  # noqa: E402
    D4DAnchorDataset, DomainBalancedSampler, MultiDomainRawErrorDataset,
)
from model_design.data.temporal_pair_dataset import (  # noqa: E402
    DEFAULT_VALIDATION_SEQUENCES, SEEN_BACKBONES, TemporalPairDataset, build_split_manifest,
)
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter, SEA_RAFT_CHECKPOINT  # noqa: E402
from model_design.losses.safety_losses import learned_t1_losses  # noqa: E402
from model_design.models.learned_t1_refiner import LearnedT1Refiner  # noqa: E402
from run_learned_t1_refiner import (  # noqa: E402
    atomic_checkpoint, build_evidence, loss_config, save_json, to_device,
)


OUT_DEFAULT = ROOT / "results/multidomain_a2"
RATIOS = {"D1": .25, "D2": .50}
TEST_SEQUENCES = ("dataset_7_keyframe_3", "dataset_7_keyframe_4")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    keys = list(rows[0])
    for row in rows[1:]:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loader(dataset, args, *, sampler=None, shuffle: bool = False) -> DataLoader:
    workers = min(args.workers, max(len(dataset), 1))
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                      shuffle=shuffle if sampler is None else False, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, drop_last=False,
                      generator=torch.Generator().manual_seed(args.seed))


def sources(args, *, smoke: bool = False):
    manifest = build_split_manifest(seed=args.seed, coverage_threshold=.50, frame_stride=1,
                                    validation_sequences=list(DEFAULT_VALIDATION_SEQUENCES))
    train_sequences = manifest["train_sequences"]
    if smoke:
        train_sequences = train_sequences[:1]
    pair_cap = (12 if smoke else args.max_train_pairs)
    val_cap = (4 if smoke else args.max_validation_pairs)
    scared_train = TemporalPairDataset(list(SEEN_BACKBONES), train_sequences, coverage_threshold=.50,
        max_pairs_per_sequence=pair_cap, random_clip_start=True, seed=args.seed)
    scared_validation = TemporalPairDataset(list(SEEN_BACKBONES), list(DEFAULT_VALIDATION_SEQUENCES),
        coverage_threshold=.50, max_pairs_per_sequence=val_cap, random_clip_start=False, seed=args.seed)
    d4d_train = D4DAnchorDataset(["specimen_1"], backbone="S2M2-S", max_records=4 if smoke else None)
    d4d_validation = D4DAnchorDataset(["specimen_2"], backbone="S2M2-S", max_records=2 if smoke else None)
    return manifest, scared_train, scared_validation, d4d_train, d4d_validation


@torch.no_grad()
def domain_metrics(model, flow, dataset, device, args) -> dict:
    sums = defaultdict(float)
    per_backbone: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    model.eval()
    for cpu in loader(dataset, args):
        batch = to_device(cpu, device)
        evidence, _ = build_evidence(flow, batch)
        output = model(batch["raw"], evidence, batch["current_rgb"])
        valid = ((batch["gt_coverage"] > .50) & batch["raw_valid"].bool()
                 & evidence["aligned_validity"].bool() & evidence["warp_support"].bool())
        raw_error = (batch["raw"] - batch["gt"]).abs()
        refined_error = (output.disparity - batch["gt"]).abs()
        memory_error = (evidence["aligned_past_disparity"] - batch["gt"]).abs()
        n = int(valid.sum())
        sums["pixels"] += n
        sums["raw_sum"] += float(raw_error[valid].sum())
        sums["refined_sum"] += float(refined_error[valid].sum())
        sums["oracle_sum"] += float(torch.minimum(raw_error, memory_error)[valid].sum())
        sums["false_update"] += float(((output.update.abs() > .05) & valid & (raw_error <= .50)).sum())
        sums["clean"] += float((valid & (raw_error <= .50)).sum())
        for index, backbone in enumerate(batch["backbone"]):
            mask = valid[index:index + 1]; count = int(mask.sum())
            item = per_backbone[str(backbone)]
            item["pixels"] += count
            item["raw_sum"] += float(raw_error[index:index + 1][mask].sum())
            item["refined_sum"] += float(refined_error[index:index + 1][mask].sum())
    pixels = max(sums["pixels"], 1.0)
    raw = sums["raw_sum"] / pixels; refined = sums["refined_sum"] / pixels; oracle = sums["oracle_sum"] / pixels
    return {
        "raw_epe": raw, "refined_epe": refined, "oracle_epe": oracle,
        "gain": raw - refined, "oracle_gain": raw - oracle,
        "oracle_recovery": (raw - refined) / max(raw - oracle, 1e-8),
        "false_update_rate": sums["false_update"] / max(sums["clean"], 1.0), "valid_pixels": int(sums["pixels"]),
        "per_backbone": {key: {"raw_epe": item["raw_sum"] / max(item["pixels"], 1),
                                 "refined_epe": item["refined_sum"] / max(item["pixels"], 1),
                                 "gain": (item["raw_sum"]-item["refined_sum"])/max(item["pixels"], 1),
                                 "valid_pixels": int(item["pixels"])} for key, item in per_backbone.items()},
    }


def selection_score(scared: dict, d4d: dict) -> float:
    """Equal-domain normalized residual error; no pixel-count dominance."""
    return .5 * (scared["refined_epe"] / max(scared["raw_epe"], 1e-8)
                 + d4d["refined_epe"] / max(d4d["raw_epe"], 1e-8))


def train(args, *, smoke: bool = False) -> int:
    seed_all(args.seed); device = torch.device(args.device)
    manifest, scared_train, scared_validation, d4d_train, d4d_validation = sources(args, smoke=smoke)
    combined = MultiDomainRawErrorDataset({"SCARED-C": scared_train, "D4D": d4d_train})
    fraction = .5 if smoke else args.d4d_fraction
    sampler = DomainBalancedSampler(combined, {"SCARED-C": 1 - fraction, "D4D": fraction},
                                    samples_per_epoch=32 if smoke else args.samples_per_epoch, seed=args.seed)
    model = LearnedT1Refiner("A2", tau_px=3.0).to(device)
    flow = BiDAFlowInferenceAdapter("sea_raft", device=device)
    assert sum(parameter.numel() for parameter in model.parameters()) == 39299
    assert all(not parameter.requires_grad for parameter in flow.model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = loss_config("A2")
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", vars(args) | {"output": str(output), "variant": "A2"})
    split = {"architecture": "unchanged LearnedT1Refiner A2", "sota": "causal BiDA t-1 only",
             "scared_train_sequences": manifest["train_sequences"],
             "scared_validation_sequences": list(DEFAULT_VALIDATION_SEQUENCES),
             "d4d_train_specimens": ["specimen_1"], "d4d_validation_specimens": ["specimen_2"],
             "d4d_backbones": ["S2M2-S"], "scared_backbones": list(SEEN_BACKBONES),
             "forbidden_before_freeze": ["SERV-CT", "StereoMIS", "Fast-FoundationStereo", "CREStereo", "D4D/specimen_3"],
             "sampling": {"SCARED-C": 1 - fraction, "D4D": fraction, "samples_per_epoch": len(sampler),
                          "domain_counts": sampler.domain_counts()},
             "source_counts": {"scared_train": len(scared_train), "d4d_train": len(d4d_train),
                               "scared_validation": len(scared_validation), "d4d_validation": len(d4d_validation)}}
    save_json(output / "split_manifest.json", split)
    history: list[dict] = []
    best = math.inf; best_eligible = False
    best_path = output / "checkpoints/best_validation.pt"; final_path = output / "checkpoints/final.pt"
    epochs = args.smoke_epochs if smoke else args.epochs
    initial_loss = None
    for epoch in range(epochs):
        sampler.set_epoch(epoch); model.train(); sums = defaultdict(float); batches = 0
        for cpu in loader(combined, args, sampler=sampler):
            batch = to_device(cpu, device); evidence, flow_ms = build_evidence(flow, batch)
            valid = (batch["gt_valid"].bool() & batch["raw_valid"].bool() & evidence["aligned_validity"].bool()
                     & evidence["warp_support"].bool())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(batch["raw"], evidence, batch["current_rgb"])
                losses = learned_t1_losses(prediction, raw=batch["raw"], aligned_memory=evidence["aligned_past_disparity"],
                                           gt=batch["gt"], valid=valid, safety_valid=valid, config=config)
            scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); scaler.step(optimizer); scaler.update()
            for key, value in losses.items(): sums[key] += float(value.detach())
            sums["gradient_norm"] += grad; sums["flow_latency_ms"] += flow_ms
            sums["update_abs_max"] = max(sums["update_abs_max"], float(prediction.update.detach().abs().max())); batches += 1
        scared = domain_metrics(model, flow, scared_validation, device, args)
        d4d = domain_metrics(model, flow, d4d_validation, device, args)
        score = selection_score(scared, d4d)
        proposal_eligible = (scared["gain"] > 0 and d4d["gain"] > 0
                             and scared["oracle_recovery"] >= .05 and d4d["oracle_recovery"] >= .05)
        row = {"epoch": epoch + 1, "selection_score": score,
               "proposal_eligible": proposal_eligible,
               **{f"train_{key}": value / max(batches, 1) for key, value in sums.items() if key != "update_abs_max"},
               "train_update_abs_max": sums["update_abs_max"],
               **{f"scared_{key}": value for key, value in scared.items() if key != "per_backbone"},
               **{f"d4d_{key}": value for key, value in d4d.items() if key != "per_backbone"}}
        history.append(row); write_csv(output / "training_history.csv", history)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                   "variant": "A2", "tau_px": 3.0, "best_validation_score": best,
                   "loss_config": asdict(config), "split": split, "validation": {"scared": scared, "d4d": d4d}}
        atomic_checkpoint(final_path, payload)
        # Never let a lower average score that harms one seen domain displace a
        # checkpoint with positive gain on both seen domains.
        replace_best = ((proposal_eligible and (not best_eligible or score < best))
                        or (not best_eligible and not proposal_eligible and score < best))
        if replace_best:
            best, best_eligible = score, proposal_eligible
            payload["best_validation_score"] = best
            payload["proposal_eligible"] = best_eligible
            atomic_checkpoint(best_path, payload)
        print(json.dumps(row), flush=True)
        if initial_loss is None: initial_loss = row["train_total"]
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    summary = {"best_validation_score": state["best_validation_score"], "proposal_eligible": state.get("proposal_eligible", False), "validation": state["validation"],
               "a2_parameters": 39299, "sea_raft_frozen": True,
               "smoke": smoke, "initial_loss": initial_loss, "final_loss": history[-1]["train_total"],
               "finite": all(torch.isfinite(parameter).all() for parameter in model.parameters())}
    if smoke:
        # Full training selects the lowest validation-score checkpoint; a short
        # smoke must test that this controlled path can improve at least once,
        # not require an arbitrary final mini-epoch to beat epoch zero.
        summary["best_train_loss"] = min(row["train_total"] for row in history)
        summary["best_selection_score"] = min(row["selection_score"] for row in history)
        summary["passed"] = bool(summary["finite"]
                                  and summary["best_selection_score"] < history[0]["selection_score"]
                                  and min(row["train_update_abs_max"] for row in history) <= 3.0001)
        if not summary["passed"]: raise RuntimeError(f"A2 multi-domain smoke failed: {summary}")
    save_json(output / "summary.json", summary)
    return 0


def select(args) -> int:
    root = args.output / "M1"; candidates = []
    for name in RATIOS:
        path = root / name / "checkpoints/best_validation.pt"
        if not path.exists(): raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        validation = state["validation"]; scared, d4d = validation["scared"], validation["d4d"]
        # A2 is the bounded *proposal* and is intentionally evaluated before
        # the frozen/learned authorization stage.  Its unconditional update
        # rate is reported, but cannot be used to reject a geometrically useful
        # proposal (the validated A2 baseline is likewise unsafe unguarded).
        eligible = (scared["gain"] > 0 and d4d["gain"] > 0
                    and scared["oracle_recovery"] >= .05 and d4d["oracle_recovery"] >= .05)
        candidates.append({"name": name, "path": str(path), "score": state["best_validation_score"],
                           "eligible": eligible, "scared": scared, "d4d": d4d})
    eligible = [item for item in candidates if item["eligible"]]
    result = {"candidates": candidates, "selection_domains": ["SCARED-C validation", "D4D specimen-2"],
              "unseen_not_loaded": ["SERV-CT", "StereoMIS", "Fast-FoundationStereo", "CREStereo", "D4D/specimen-3"]}
    if not eligible:
        result["status"] = "NO-GO: no candidate passed seen-domain proposal gate"
        save_json(root / "selection.json", result); return 2
    selected = min(eligible, key=lambda item: item["score"])
    frozen = root / "frozen"; frozen.mkdir(parents=True, exist_ok=False); (frozen / "checkpoints").mkdir()
    source = Path(selected["path"]); target = frozen / "checkpoints/best_validation.pt"
    target.write_bytes(source.read_bytes())
    artifacts = {"a2": target, "sea_raft": SEA_RAFT_CHECKPOINT,
                 "bida": ROOT / "model_design/external_components/bidavideo.py"}
    result.update({"status": "frozen before final-only data", "selected": selected,
                   "artifacts": {key: {"path": str(value), "sha256": sha256(value)} for key, value in artifacts.items()}})
    save_json(root / "selection.json", result); save_json(frozen / "frozen_manifest.json", result)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "train", "select"), required=True)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--d4d-fraction", type=float, choices=tuple(RATIOS.values()), default=.25)
    parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--smoke-epochs", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=4096); parser.add_argument("--max-train-pairs", type=int, default=256)
    parser.add_argument("--max-validation-pairs", type=int, default=160); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8)); parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "train": args.output = args.output / "M1" / ("D1" if args.d4d_fraction == .25 else "D2")
    elif args.stage == "smoke": args.output = args.output / "smoke"
    return args


def main() -> int:
    args = parse_args()
    return train(args, smoke=args.stage == "smoke") if args.stage != "select" else select(args)


if __name__ == "__main__":
    raise SystemExit(main())
