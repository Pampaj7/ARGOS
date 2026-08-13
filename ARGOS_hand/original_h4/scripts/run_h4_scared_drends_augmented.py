#!/usr/bin/env python3
"""Locked H4 training: SCARED-C plus DRENDS, with no held-out access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_h4_augmented_fusion_probe import (  # noqa: E402
    CACHE_WIDTH, H4_CANONICAL_CHECKPOINT, NATIVE_WIDTH, SEEN_BACKBONES, BiDAFlowInferenceAdapter,
    CODDStyleFusionHead, FrozenResNet18Layer1, RESNET18_CHECKPOINT, atomic, atomic_copy, codd_config,
    frame, run_clip, run_clip_placeholder, save_json, seed_all, sha256, should_stop, to_device,
)
from model_design.data.drends_temporal_dataset import DrendsTemporalClipDataset, build_raft_cache  # noqa: E402
from model_design.data.temporal_clip_dataset import TemporalClipDataset  # noqa: E402

SCARED_TRAIN = ("dataset_1_keyframe_1", "dataset_1_keyframe_2", "dataset_1_keyframe_3", "dataset_3_keyframe_1", "dataset_3_keyframe_2", "dataset_3_keyframe_3", "dataset_3_keyframe_4", "dataset_6_keyframe_1", "dataset_6_keyframe_2", "dataset_6_keyframe_3", "dataset_6_keyframe_4", "dataset_2_keyframe_2", "dataset_2_keyframe_3")
SCARED_VAL = ("dataset_2_keyframe_4",)
DRENDS_TRAIN = ("Vid10_Liver_Med", "Vid12_Pancreas_Ext")
DRENDS_VAL = ("Vid13_Pancreas_Med",)
DRENDS_HELDOUT = "Vid14_Pancreas_High"
OUT = ROOT / "results/h4_scared_drends_augmented/seed_0"
FINAL_DIR = ROOT / "model_design/checkpoints/h4_scared_drends_augmented"
CACHE = ROOT / "cache_drends_backbones"
STEPS_PER_EPOCH = 252


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("train", "dry-run"), default="train")
    p.add_argument("--output", type=Path, default=OUT); p.add_argument("--cache-root", type=Path, default=CACHE)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--seed", type=int, default=0); p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=10); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--workers", type=int, default=20); p.add_argument("--preload-workers", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=2e-4); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--clip-length", type=int, default=4)
    p.add_argument("--coverage-threshold", type=float, default=.50); p.add_argument("--tau-reset-native-px", type=float, default=5.0); p.add_argument("--tau-fusion-native-px", type=float, default=1.0); p.add_argument("--alpha-reg", type=float, default=.2)
    p.add_argument("--memory-state", default="recurrent"); p.add_argument("--disable-learned-stereo-evidence", action="store_true")
    return p.parse_args()


def identity(c): return {k: str(v) if isinstance(v, Path) else v for k, v in vars(c).items() if k not in {"mode", "output"}}


def validate(c):
    locked = {"seed": 0, "epochs": 150, "patience": 10, "batch_size": 32, "workers": 20, "preload_workers": 20, "learning_rate": 2e-4, "weight_decay": 1e-4, "clip_length": 4, "coverage_threshold": .50, "tau_reset_native_px": 5., "tau_fusion_native_px": 1., "alpha_reg": .2}
    bad = {k: (getattr(c, k), v) for k, v in locked.items() if getattr(c, k) != v}
    if bad: raise ValueError(f"locked h4_scared_drends_augmented mismatch: {bad}")
    if c.device not in {"cuda:0", "cpu"}: raise ValueError("only physical GPU0 or CPU preflight is allowed")
    if c.memory_state != "recurrent" or c.disable_learned_stereo_evidence: raise ValueError("H4 recurrence and learned stereo evidence are locked")
    if set(SCARED_TRAIN) & set(SCARED_VAL) or set(DRENDS_TRAIN) & set(DRENDS_VAL) or DRENDS_HELDOUT in DRENDS_TRAIN + DRENDS_VAL: raise RuntimeError("split leakage")


def split(c):
    return {"scared_train": list(SCARED_TRAIN), "scared_validation": list(SCARED_VAL), "drends_train": list(DRENDS_TRAIN), "drends_validation": list(DRENDS_VAL), "drends_heldout": DRENDS_HELDOUT,
            "heldout_note": "Vid14 is historical heldout/comparison, not a pristine project-wide test", "d7_unopened_until_gate": True, "drends_tof_caveat": "temporally smoothed ToF reference is nonindependent", "grid": [144, 180], "disparity_scale": "180/1280"}


def validation_macro(scared, drends): return .5 * (scared["ratio"] + drends["ratio"])


class FixedMixedBatchSampler(Sampler[list[int]]):
    """252 deterministic batches: 8 from each SCARED source and 8 DRENDS."""
    def __init__(self, scared, drends, batch_size=32, seed=0, steps=STEPS_PER_EPOCH):
        if batch_size != 32: raise ValueError("locked batch size 32")
        self.scared, self.drends, self.seed, self.steps, self.epoch = scared, drends, seed, steps, 0
        self.groups = {name: [i for i, r in enumerate(scared.records) if r.backbone == name] for name in SEEN_BACKBONES}
        if any(not v for v in self.groups.values()) or not len(drends): raise RuntimeError("empty mixed sampling source")
    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return self.steps
    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch * 1009)
        for _ in range(self.steps):
            batch = []
            for source in SEEN_BACKBONES:
                picks = torch.randint(len(self.groups[source]), (8,), generator=g).tolist(); batch += [self.groups[source][i] for i in picks]
            picks = torch.randint(len(self.drends), (8,), generator=g).tolist(); batch += [len(self.scared) + i for i in picks]
            yield batch


def make_scared(sequences, c): return TemporalClipDataset(SEEN_BACKBONES, sequences, clip_length=c.clip_length, clip_stride=c.clip_length, coverage_threshold=c.coverage_threshold, include_right_rgb=True, seed=c.seed)


def loader(dataset, c, sampler=None):
    common = {"num_workers": c.workers, "persistent_workers": c.workers > 0, "pin_memory": True, "prefetch_factor": 4 if c.workers else None}
    return DataLoader(dataset, batch_sampler=sampler, **common) if sampler else DataLoader(dataset, batch_size=1, **common)


def evaluate_domain(model, extractor, adapter, data, c, name):
    model.eval(); total = {"raw": 0., "fused": 0., "count": 0, "harmful": 0., "depth_sq_raw": 0., "depth_sq_fused": 0., "depth_count": 0}
    with torch.no_grad():
        for cpu in loader(data, c):
            _, states = run_clip(model, extractor, adapter, to_device(cpu, next(model.parameters()).device), c, training=False)
            for state in states:
                item, output, mask = state["item"], state["output"], state["raw_memory_valid"]
                if not bool(mask.any()): continue
                raw_err, fused_err = (item["raw"] - item["gt"]).abs(), (output.fused_disparity - item["gt"]).abs(); n = int(mask.sum())
                total["raw"] += float(raw_err[mask].sum()); total["fused"] += float(fused_err[mask].sum()); total["count"] += n; total["harmful"] += int((mask & (fused_err > raw_err + .1)).sum())
                if name == "drends":
                    product, gt_depth = item["focal_baseline_mm"].view(-1,1,1,1), item["gt_depth_mm"]
                    raw_depth, fused_depth = product / item["raw"].clamp_min(1e-6), product / output.fused_disparity.clamp_min(1e-6)
                    total["depth_sq_raw"] += float(((raw_depth - gt_depth).square()[mask]).sum()); total["depth_sq_fused"] += float(((fused_depth - gt_depth).square()[mask]).sum()); total["depth_count"] += n
    if not total["count"]: raise RuntimeError(f"empty {name} validation")
    raw, fused = total["raw"] / total["count"], total["fused"] / total["count"]
    result = {"raw_epe": raw, "fused_epe": fused, "ratio": fused / raw, "gain": raw - fused, "valid_count": total["count"], "harmful_update_rate": total["harmful"] / total["count"]}
    if name == "drends": result |= {"raw_depth_rmse_mm": math.sqrt(total["depth_sq_raw"] / total["depth_count"]), "fused_depth_rmse_mm": math.sqrt(total["depth_sq_fused"] / total["depth_count"])}
    return result


def save_state(path, payload): atomic(path, payload)


def train(c):
    validate(c); c.output.mkdir(parents=True, exist_ok=True)
    final = FINAL_DIR / "best_validation.pt"
    if final.exists() or (c.output / "checkpoints/last.pt").exists(): raise FileExistsError("collision/resume refused for fresh locked run")
    save_json(c.output / "configuration.json", vars(c)); audit = split(c); save_json(c.output / "split_audit.json", audit); save_json(c.output / "status.json", {"state":"caching", "pid":os.getpid(), "updated_unix":time.time()})
    cache_report = build_raft_cache(c.cache_root, DRENDS_TRAIN + DRENDS_VAL, c.device); save_json(c.output / "cache_summary.json", cache_report)
    seed_all(c.seed); device = torch.device(c.device)
    scared, drends = make_scared(SCARED_TRAIN, c), DrendsTemporalClipDataset(DRENDS_TRAIN, c.cache_root, clip_length=c.clip_length, coverage_threshold=c.coverage_threshold)
    scared_val, drends_val = make_scared(SCARED_VAL, c), DrendsTemporalClipDataset(DRENDS_VAL, c.cache_root, clip_length=c.clip_length, coverage_threshold=c.coverage_threshold)
    preload = {"scared_train": scared.pairs.preload_frame_data(c.preload_workers), "drends_train": drends.preload_frame_data(c.preload_workers), "scared_validation": scared_val.pairs.preload_frame_data(c.preload_workers), "drends_validation": drends_val.preload_frame_data(c.preload_workers)}; save_json(c.output / "preload_summary.json", preload)
    mixer = FixedMixedBatchSampler(scared, drends, c.batch_size, c.seed); train_loader = loader(ConcatDataset((scared, drends)), c, mixer)
    extractor, adapter = FrozenResNet18Layer1().to(device), BiDAFlowInferenceAdapter("sea_raft", device=device); assert not any(p.requires_grad for p in extractor.parameters()) and not any(p.requires_grad for p in adapter.model.parameters())
    model = optimizer = None; scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"); best, bad, history = math.inf, 0, []
    for epoch in range(c.epochs):
        mixer.set_epoch(epoch); totals = 0.
        for batch, cpu in enumerate(train_loader, 1):
            clip = to_device(cpu, device)
            if model is None:
                with torch.no_grad(): _, channels = run_clip_placeholder(extractor, adapter, clip, c)
                model = CODDStyleFusionHead(channels).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
            model.train()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"): loss, _ = run_clip(model, extractor, adapter, clip, c, training=True)
            if not math.isfinite(float(loss.detach())): raise RuntimeError("non-finite train loss")
            optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update(); totals += float(loss.detach())
            if batch == 1 or batch % 10 == 0: print(json.dumps({"event":"train_batch", "epoch":epoch+1, "batch":batch, "loss":float(loss.detach())}), flush=True)
        sc, dr = evaluate_domain(model, extractor, adapter, scared_val, c, "scared"), evaluate_domain(model, extractor, adapter, drends_val, c, "drends")
        macro = validation_macro(sc, dr); row = {"epoch":epoch+1, "train_total":totals/STEPS_PER_EPOCH, "validation_macro_ratio":macro, **{f"scared_{k}":v for k,v in sc.items()}, **{f"drends_{k}":v for k,v in dr.items()}}; history.append(row)
        with (c.output / "training_history.csv").open("w", newline="") as f: w=csv.DictWriter(f, fieldnames=history[0]); w.writeheader(); w.writerows(history)
        if not math.isfinite(macro): raise RuntimeError("non-finite validation macro")
        improved = macro < best; best, bad = (macro, 0) if improved else (best, bad+1)
        payload = {"model":model.state_dict(), "optimizer":optimizer.state_dict(), "scaler":scaler.state_dict(), "epoch":epoch+1, "best":best, "bad_epochs":bad, "cue_channels":model.full[0].in_channels, "config":vars(c), "config_identity":identity(c), "split":audit, "resnet_sha256":sha256(RESNET18_CHECKPOINT), "selection_metric":"0.5*(scared fused/raw EPE + drends fused/raw EPE)"}
        save_state(c.output / "checkpoints/last.pt", payload)
        if improved: save_state(c.output / "checkpoints/best_validation.pt", payload)
        save_json(c.output / "status.json", {"state":"training", "pid":os.getpid(), "epoch":epoch+1, "best_macro_ratio":best, "updated_unix":time.time()}); print(json.dumps(row), flush=True)
        if should_stop(c.patience, bad): break
    state = torch.load(c.output / "checkpoints/best_validation.pt", map_location="cpu", weights_only=False); model = CODDStyleFusionHead(state["cue_channels"]).to(device); model.load_state_dict(state["model"])
    sc, dr = evaluate_domain(model, extractor, adapter, scared_val, c, "scared"), evaluate_domain(model, extractor, adapter, drends_val, c, "drends")
    gate = sc["ratio"] <= 1 and dr["ratio"] <= 1 and dr["fused_depth_rmse_mm"] <= dr["raw_depth_rmse_mm"]
    summary = {"selection": {"scared":sc, "drends":dr, "macro_ratio":validation_macro(sc, dr)}, "gate": {"passed":gate, "requirements":"ratio<=1 on both validation domains and DRENDS depth RMSE<=raw", "heldout_access": "forbidden unless passed"}}
    save_json(c.output / "validation_summary.json", summary)
    if not gate:
        save_json(c.output / "status.json", {"state":"no_go", "pid":os.getpid(), **summary}); raise RuntimeError("NO-GO validation gate; D7/Vid14 remain unopened")
    FINAL_DIR.mkdir(parents=True, exist_ok=True); atomic_copy(c.output / "checkpoints/best_validation.pt", final)
    provenance = {"profile":"h4_scared_drends_augmented", "checkpoint":str(final), "checkpoint_sha256":sha256(final), "configuration":vars(c), "split":audit, "validation":summary, "runner":str(Path(__file__)), "runner_sha256":sha256(Path(__file__)), "canonical_h4_checkpoint_sha256":sha256(H4_CANONICAL_CHECKPOINT)}
    save_json(FINAL_DIR / "provenance.json", provenance); save_json(FINAL_DIR / "configuration.json", vars(c)); save_json(FINAL_DIR / "split_audit.json", audit); shutil.copy2(c.output / "training_history.csv", FINAL_DIR / "training_history.csv"); save_json(c.output / "status.json", {"state":"complete", "pid":os.getpid(), **summary})


def main():
    c=arguments(); validate(c)
    if c.mode == "dry-run": print(json.dumps({"config":vars(c), "split":split(c), "sampling":{"steps":STEPS_PER_EPOCH,"scared":.75,"drends":.25,"each_scared_backbone":.25}}, default=str)); return
    train(c)
if __name__ == "__main__": main()
