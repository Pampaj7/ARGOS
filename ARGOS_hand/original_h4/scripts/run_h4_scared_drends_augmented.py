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
import tempfile
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
from model_design.comparison.drends_evaluation import _prediction_depth_mm  # noqa: E402
from model_design.metrics.unified_metrics import MetricConfig  # noqa: E402

# Match the accepted SCARED-C H4 augmented split.  keyframe_1 was quality-gate
# rejected and has no curated manifest/cache.
SCARED_TRAIN = ("dataset_1_keyframe_2", "dataset_1_keyframe_3", "dataset_3_keyframe_1", "dataset_3_keyframe_2", "dataset_3_keyframe_3", "dataset_3_keyframe_4", "dataset_6_keyframe_1", "dataset_6_keyframe_2", "dataset_6_keyframe_3", "dataset_6_keyframe_4", "dataset_2_keyframe_2", "dataset_2_keyframe_3")
SCARED_VAL = ("dataset_2_keyframe_4",)
DRENDS_TRAIN = ("Vid10_Liver_Med", "Vid12_Pancreas_Ext")
DRENDS_VAL = ("Vid13_Pancreas_Med",)
DRENDS_HELDOUT = "Vid14_Pancreas_High"
OUT = ROOT / "results/h4_scared_drends_augmented/seed_0"
FINAL_DIR = ROOT / "model_design/checkpoints/h4_scared_drends_augmented"
CACHE = ROOT / "cache_drends_backbones"
STEPS_PER_EPOCH = 252
EXPECTED_BEST_EPOCH = 27
EXPECTED_BEST_CHECKPOINT_SHA256 = "e90586fdb90305c37887f191e111a4e4dd9b2c8ba0c5a7c951e4e4d797fd20b4"
EXPECTED_TRAINING_SOURCE_SNAPSHOT_SHA256 = "bb03e605791bb8cd9aac5a3c21b27049692c271a696b6e88926ca54b073f1c0d"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("train", "revalidate", "dry-run"), default="train")
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
    metrics = MetricConfig()
    with torch.no_grad():
        for cpu in loader(data, c):
            _, states = run_clip(model, extractor, adapter, to_device(cpu, next(model.parameters()).device), c, training=False)
            for state in states:
                item, output = state["item"], state["output"]
                support = item["gt_valid"].bool()
                if name == "drends":
                    # Match frozen DRENDS evaluation: fixed GT support, clipped
                    # positive depths, and the official invalid-prediction penalty.
                    product = float(item["focal_baseline_mm"].reshape(-1)[0])
                    depth_support = support.detach().cpu().numpy()
                    gt_depth = item["gt_depth_mm"].detach().cpu().numpy()
                    raw_depth = _prediction_depth_mm(item["raw"].detach().cpu().numpy(), product)
                    fused_depth = _prediction_depth_mm(output.fused_disparity.detach().cpu().numpy(), product)
                    penalty = metrics.invalid_penalty_mm
                    raw_error = np.where(np.isfinite(raw_depth) & (raw_depth > 0), raw_depth - gt_depth, penalty)
                    fused_error = np.where(np.isfinite(fused_depth) & (fused_depth > 0), fused_depth - gt_depth, penalty)
                    total["depth_sq_raw"] += float(np.square(raw_error[depth_support]).sum()); total["depth_sq_fused"] += float(np.square(fused_error[depth_support]).sum()); total["depth_count"] += int(depth_support.sum())
                if not bool(support.any()): continue
                raw_valid, fused_valid = torch.isfinite(item["raw"]) & (item["raw"] > 0), torch.isfinite(output.fused_disparity) & (output.fused_disparity > 0)
                raw_err = torch.where(raw_valid, (item["raw"] - item["gt"]).abs(), torch.full_like(item["raw"], metrics.invalid_penalty_px))
                fused_err = torch.where(fused_valid, (output.fused_disparity - item["gt"]).abs(), torch.full_like(output.fused_disparity, metrics.invalid_penalty_px)); n = int(support.sum())
                total["raw"] += float(raw_err[support].sum()); total["fused"] += float(fused_err[support].sum()); total["count"] += n; total["harmful"] += int((support & (fused_err > raw_err + .1)).sum())
    if not total["count"]: raise RuntimeError(f"empty {name} validation")
    raw, fused = total["raw"] / total["count"], total["fused"] / total["count"]
    result = {"raw_epe": raw, "fused_epe": fused, "ratio": fused / raw, "gain": raw - fused, "valid_count": total["count"], "harmful_update_rate": total["harmful"] / total["count"]}
    if name == "drends":
        if not total["depth_count"]: raise RuntimeError("empty DRENDS depth validation support")
        result |= {"raw_depth_rmse_mm": math.sqrt(total["depth_sq_raw"] / total["depth_count"]), "fused_depth_rmse_mm": math.sqrt(total["depth_sq_fused"] / total["depth_count"]), "depth_valid_count": total["depth_count"]}
    return result


def evaluate_scared_d2(model, extractor, c):
    """Use the frozen D2 driver so its strict all-anchor mask stays authoritative."""
    from model_design.comparison.run_comparison import _scared
    bundles = []
    class Adapter:
        horizon = 4
        def start(self, frame): return {"disparity": frame["raw"], "support": frame["raw_valid"].bool(), "reset": True, "state_age": 0, "diagnostics": {"update_magnitude": 0.0}}
        def step(self, frame):
            from model_design.external_components.bidavideo import temporal_disparity_evidence
            from model_design.models.codd_style_fusion import build_codd_cues
            with torch.inference_mode():
                evidence = temporal_disparity_evidence(frame["raw"], frame["past_disparity"], frame["forward_flow"], frame["backward_flow"], current_valid=frame["raw_valid"], past_valid=frame["past_valid"], current_rgb=frame["current_rgb"], past_rgb=frame["past_rgb"])
                cues = build_codd_cues(extractor, raw=frame["raw"], aligned_memory=evidence.aligned_past_disparity, current_rgb=frame["current_rgb"], current_right_rgb=frame["current_right_rgb"], past_rgb=frame["past_rgb"], flow_current_to_past=frame["forward_flow"], flow_magnitude=evidence.flow_magnitude, forward_backward_confidence=evidence.forward_backward_confidence, warp_support=evidence.warp_support, aligned_valid=evidence.aligned_validity, include_learned_stereo_evidence=True)
                output = model(cues, frame["raw"], evidence.aligned_past_disparity); support = frame["raw_valid"].bool() & evidence.aligned_validity.bool() & evidence.warp_support.bool()
            return {"disparity": output.fused_disparity, "support": support, "reset": bool(frame["reanchor"]), "state_age": int(frame["state_age"]), "diagnostics": {"update_magnitude": float((output.fused_disparity - frame["raw"]).abs()[support].mean()) if bool(support.any()) else 0.0}}
    _scared(argparse.Namespace(dataset="scared-d2", sequences=list(SCARED_VAL), backbones=list(SEEN_BACKBONES), max_frames=None, smoke=False, device=c.device, flow_batch_size=c.batch_size), Adapter(), bundles.append)
    total = {"raw": 0., "fused": 0., "count": 0, "harmful": 0}; penalty = MetricConfig().invalid_penalty_px
    for bundle in bundles:
        support = np.asarray(bundle["gt_valid"], bool) & np.asarray(bundle["protocol_mask"], bool)
        raw, fused, gt = np.asarray(bundle["raw_disparity"]), np.asarray(bundle["refined_disparity"]), np.asarray(bundle["gt_disparity"])
        raw_error = np.where(np.isfinite(raw) & (raw > 0), np.abs(raw - gt), penalty); fused_error = np.where(np.isfinite(fused) & (fused > 0), np.abs(fused - gt), penalty)
        total["raw"] += float(raw_error[support].sum()); total["fused"] += float(fused_error[support].sum()); total["count"] += int(support.sum()); total["harmful"] += int((support & (fused_error > raw_error + .1)).sum())
    if not total["count"]: raise RuntimeError("empty strict D2 validation")
    raw, fused = total["raw"] / total["count"], total["fused"] / total["count"]
    return {"raw_epe": raw, "fused_epe": fused, "ratio": fused / raw, "gain": raw - fused, "valid_count": total["count"], "harmful_update_rate": total["harmful"] / total["count"], "protocol": "paper_d2_strict_all_anchors"}


def save_state(path, payload): atomic(path, payload)


def publish_final_bundle(c, audit, summary, final):
    """Publish checkpoint and required provenance as one directory rename."""
    if final.exists(): raise FileExistsError(f"refusing to overwrite final artifact: {final}")
    source = c.output / "checkpoints/best_validation.pt"
    if not source.is_file(): raise FileNotFoundError(f"missing best validation checkpoint: {source}")
    final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=final.parent, prefix=f".{final.name}.") as temporary:
        stage = Path(temporary)
        checkpoint = stage / "best_validation.pt"; shutil.copy2(source, checkpoint)
        if sha256(source) != sha256(checkpoint): raise RuntimeError("final checkpoint copy hash mismatch")
        dataset = ROOT / "model_design/data/drends_temporal_dataset.py"
        provenance = {"profile":"h4_scared_drends_augmented", "checkpoint":str(final / "best_validation.pt"), "checkpoint_sha256":sha256(checkpoint), "configuration":vars(c), "configuration_identity":identity(c), "split":audit, "validation":summary, "runner":str(Path(__file__)), "runner_sha256":sha256(Path(__file__)), "drends_dataset":str(dataset), "drends_dataset_sha256":sha256(dataset), "canonical_h4_checkpoint_sha256":sha256(H4_CANONICAL_CHECKPOINT)}
        if "revalidation" in summary: provenance["revalidation"] = summary["revalidation"]
        for name, value in (("provenance.json", provenance), ("configuration.json", vars(c)), ("split_audit.json", audit)):
            save_json(stage / name, value)
        shutil.copy2(c.output / "training_history.csv", stage / "training_history.csv")
        os.replace(stage, final)


def train(c):
    validate(c); c.output.mkdir(parents=True, exist_ok=True)
    final = FINAL_DIR / "best_validation.pt"
    if final.exists() or (c.output / "checkpoints/last.pt").exists(): raise FileExistsError("collision/resume refused for fresh locked run")
    save_json(c.output / "configuration.json", vars(c)); audit = split(c); save_json(c.output / "split_audit.json", audit); save_json(c.output / "status.json", {"state":"caching", "pid":os.getpid(), "updated_unix":time.time()})
    try:
        cache_report = build_raft_cache(c.cache_root, DRENDS_TRAIN + DRENDS_VAL, c.device)
    except Exception as error:
        save_json(c.output / "status.json", {"state":"failed", "phase":"caching", "pid":os.getpid(), "error_type":type(error).__name__, "error":str(error), "updated_unix":time.time()})
        raise
    save_json(c.output / "cache_summary.json", cache_report)
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
    publish_final_bundle(c, audit, summary, FINAL_DIR)
    save_json(c.output / "status.json", {"state":"complete", "pid":os.getpid(), **summary})


def revalidation_source(c):
    source, snapshot = c.output / "checkpoints/best_validation.pt", c.output / "source_snapshot.sha256"
    if not source.is_file() or not snapshot.is_file(): raise FileNotFoundError("revalidation requires saved best_validation.pt and source_snapshot.sha256")
    if sha256(source) != EXPECTED_BEST_CHECKPOINT_SHA256 or sha256(snapshot) != EXPECTED_TRAINING_SOURCE_SNAPSHOT_SHA256: raise RuntimeError("revalidation source hash mismatch")
    state, audit = torch.load(source, map_location="cpu", weights_only=False), split(c)
    if state.get("epoch") != EXPECTED_BEST_EPOCH or state.get("config_identity") != identity(c) or state.get("split") != audit: raise RuntimeError("saved checkpoint epoch/configuration/split mismatch")
    return state, audit, source, snapshot


def revalidate(c):
    """Re-run only D2-KF4 and Vid13 against the saved best checkpoint."""
    validate(c)
    if FINAL_DIR.exists(): raise FileExistsError(f"refusing to overwrite final artifact: {FINAL_DIR}")
    state, audit, source, snapshot = revalidation_source(c)
    save_json(c.output / "status.json", {"state":"revalidating", "pid":os.getpid(), "updated_unix":time.time()})
    device = torch.device(c.device); seed_all(c.seed)
    drends = DrendsTemporalClipDataset(DRENDS_VAL, c.cache_root, clip_length=c.clip_length, coverage_threshold=c.coverage_threshold)
    preload = {"drends_validation": drends.preload_frame_data(c.preload_workers)}
    extractor, adapter = FrozenResNet18Layer1().to(device), BiDAFlowInferenceAdapter("sea_raft", device=device)
    model = CODDStyleFusionHead(state["cue_channels"]).to(device); model.load_state_dict(state["model"])
    sc, dr = evaluate_scared_d2(model, extractor, c), evaluate_domain(model, extractor, adapter, drends, c, "drends")
    gate = sc["ratio"] <= 1 and dr["ratio"] <= 1 and dr["fused_depth_rmse_mm"] <= dr["raw_depth_rmse_mm"]
    revalidation = {"training_checkpoint": {"path": str(source), "sha256": sha256(source), "epoch": state.get("epoch")}, "training_source_snapshot": {"path": str(snapshot), "sha256": sha256(snapshot)}, "source": {"scared_validation": list(SCARED_VAL), "drends_validation": list(DRENDS_VAL), "preload": preload}, "protocol": {"d2": "dataset_2_keyframe_4", "drends": "Vid13_Pancreas_Med", "support": "gt_valid AND protocol_mask (identical here)", "disparity": "fixed support; MetricConfig.invalid_penalty_px", "depth": "model_design.comparison.drends_evaluation._prediction_depth_mm; fixed gt_valid support; MetricConfig.invalid_penalty_mm"}}
    summary = {"selection": {"scared": sc, "drends": dr, "macro_ratio": validation_macro(sc, dr)}, "gate": {"passed": gate, "requirements": "ratio<=1 on both validation domains and DRENDS depth RMSE<=raw", "heldout_access": "forbidden unless passed"}, "revalidation": revalidation}
    save_json(c.output / "revalidation_summary.json", summary)
    if not gate:
        save_json(c.output / "status.json", {"state":"no_go", "pid":os.getpid(), **summary}); raise RuntimeError("NO-GO validation gate; D7/Vid14 remain unopened")
    publish_final_bundle(c, audit, summary, FINAL_DIR)
    save_json(c.output / "status.json", {"state":"complete", "pid":os.getpid(), **summary}); return True


def main():
    c=arguments(); validate(c)
    if c.mode == "dry-run": print(json.dumps({"config":vars(c), "split":split(c), "sampling":{"steps":STEPS_PER_EPOCH,"scared":.75,"drends":.25,"each_scared_backbone":.25}}, default=str)); return
    try:
        (revalidate if c.mode == "revalidate" else train)(c)
    except Exception as error:
        if str(error).startswith("NO-GO validation gate"): raise
        # Cache failures are already labelled more precisely; this covers setup,
        # preload, train, validation, and publication failures after caching.
        status = c.output / "status.json"
        phase = "post_cache"
        if status.is_file():
            try: phase = json.loads(status.read_text()).get("phase", "post_cache")
            except json.JSONDecodeError: phase = "post_cache"
        save_json(status, {"state":"failed", "phase":phase, "pid":os.getpid(), "error_type":type(error).__name__, "error":str(error), "updated_unix":time.time()})
        raise
if __name__ == "__main__": main()
