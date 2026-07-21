#!/usr/bin/env python3
"""Massive, training-only ARGOS v2 causal raw-versus-memory utility selector.

No disparity proposal is trained here.  The only learned operation selects
between a cached raw disparity and the canonical causally warped t-1 memory.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V2_ROOT), str(V2_ROOT / "scripts")]

from argos_v2.sequences import accepted_sequences  # noqa: E402
from model_design.data.raw_error_dataset import CALIBRATION_SEQUENCES, TEST_SEQUENCES  # noqa: E402
from model_design.data.utility_memory_selector_dataset import (  # noqa: E402
    BalancedSequenceSampler, HierarchicalDatasetSequenceSampler,
    UtilityMemorySelectorDataset, dataset_id_from_sequence, utility_targets,
)
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter, SEA_RAFT_CHECKPOINT, temporal_disparity_evidence,
)
from model_design.external_components.stereo_photometric import stereo_photometric_evidence  # noqa: E402
from model_design.external_components.stereo_matching_evidence import (  # noqa: E402
    candidate_conditioned_stereo_matching_evidence, stereo_matching_feature_channels,
)
from model_design.losses.utility_memory_selector_losses import (  # noqa: E402
    UtilitySelectorLossConfig, utility_selector_losses,
)
from model_design.models.utility_memory_selector import (  # noqa: E402
    UtilityMemorySelector, UtilitySelectorEvidence, memory_authorization, select_raw_or_memory,
    utility_risk_authorization,
)
from run_raw_error_abstention import aggregate_rows, boundary_mask_tensor, map_metrics  # noqa: E402


SEEN_BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")
UNSEEN_BACKBONES = ("Fast-FoundationStereo", "CREStereo")
COVERAGE_THRESHOLDS = (0.05, 0.25, 0.50, 0.90)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "overfit", "train", "calibrate", "evaluate", "unseen", "summarize", "full"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    p.add_argument("--preload-workers", type=int, default=0, help="RAM-preload universal RGB/GT frames; writes no cache")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--coverage-threshold", type=float, default=.50)
    p.add_argument("--epsilon", type=float, default=.10)
    p.add_argument("--regional-target-kernel", type=int, default=1,
                   help="Odd GT-only utility pooling kernel used only while training; 1 is pixel supervision.")
    p.add_argument("--selective-target-coverage", type=float, default=.02)
    p.add_argument("--selective-coverage-weight", type=float, default=32.)
    p.add_argument("--selective-risk-weight", type=float, default=10.)
    p.add_argument("--objective", choices=("legacy", "utility_risk", "utility_calibrated", "utility_weighted", "selective_utility"), default="legacy")
    p.add_argument("--crop-height", type=int, default=96)
    p.add_argument("--crop-width", type=int, default=120)
    p.add_argument("--validation-pair-cap", type=int, default=0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--unseen-backbones", nargs="+", default=list(UNSEEN_BACKBONES))
    p.add_argument("--stereo-photometric", action="store_true",
                   help="append frozen current-frame stereo reprojection evidence; requires current right RGB")
    p.add_argument("--stereo-photometric-kernel", type=int, default=21,
                   help="odd local photometric window; selected only on validation")
    p.add_argument("--stereo-matching-evidence", choices=("none", "cost", "shape", "full"), default="none",
                   help="append frozen candidate-conditioned census cost evidence; input-only and never a metric mask")
    p.add_argument("--sampler", choices=("sequence", "hierarchical_dataset"), default="sequence",
                   help="training input order only; never a model input")
    p.add_argument("--train-sequences", nargs="+", default=None,
                   help="explicit training sequences; must be disjoint from validation/test")
    p.add_argument("--validation-sequences", nargs="+", default=None,
                   help="explicit calibration sequences; must be disjoint from train/test")
    p.add_argument("--test-sequences", nargs="+", default=None,
                   help="explicit frozen final-test sequences; must be disjoint from train/validation")
    p.add_argument("--strict-dataset-id-disjoint", action=argparse.BooleanOptionalAction, default=False,
                   help="require acquisition/session IDs to be disjoint across train, validation, and test")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def clean(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    keys: list[str] = []
    for row in rows: keys.extend(k for k in row if k not in keys)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def frozen_hashes() -> dict:
    return {
        "sea_raft_checkpoint": sha256(SEA_RAFT_CHECKPOINT),
        "bida_source": sha256(V2_ROOT / "model_design/external_components/bidavideo.py"),
        "cache_contract": "positive-left disparity [T,144,180], float16; valid mask; frame IDs verified by loader",
    }


def _validated_split(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    accepted = set(accepted_sequences())
    default_held = set(CALIBRATION_SEQUENCES) | set(TEST_SEQUENCES)
    train = list(args.train_sequences) if args.train_sequences is not None else [s for s in accepted_sequences() if s not in default_held]
    validation = list(args.validation_sequences) if args.validation_sequences is not None else list(CALIBRATION_SEQUENCES)
    test = list(args.test_sequences) if args.test_sequences is not None else list(TEST_SEQUENCES)
    roles = {"train": train, "validation": validation, "test": test}
    for role, values in roles.items():
        if not values:
            raise ValueError(f"{role} split cannot be empty")
        if len(values) != len(set(values)):
            raise ValueError(f"{role} split contains duplicate sequences")
        unknown = set(values) - accepted
        if unknown:
            raise ValueError(f"{role} contains non-accepted sequences: {sorted(unknown)}")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = set(roles[left]) & set(roles[right])
        if shared:
            raise ValueError(f"sequence leakage between {left} and {right}: {sorted(shared)}")
    if args.strict_dataset_id_disjoint:
        ids = {role: {dataset_id_from_sequence(sequence) for sequence in values} for role, values in roles.items()}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            shared = ids[left] & ids[right]
            if shared:
                raise ValueError(f"dataset-ID leakage between {left} and {right}: {sorted(shared)}")
    return train, validation, test


def manifest(args: argparse.Namespace) -> dict:
    train, validation, test = _validated_split(args)
    def pairs(seq): return sum(len(UtilityMemorySelectorDataset([SEEN_BACKBONES[0]], [s], selection_only=True)) for s in seq)
    return {
        "schema_version": 2, "split_unit": "complete sequence", "seed": args.seed,
        "train_sequences": train, "validation_sequences": validation,
        "final_seen_test_sequences": test, "seen_backbones": list(SEEN_BACKBONES),
        "unseen_backbones": list(UNSEEN_BACKBONES),
        "pairs_per_backbone": {"train": pairs(train), "validation": pairs(validation), "test": pairs(test)},
        "dataset_ids": {"train": sorted({dataset_id_from_sequence(s) for s in train}),
                        "validation": sorted({dataset_id_from_sequence(s) for s in validation}),
                        "test": sorted({dataset_id_from_sequence(s) for s in test})},
        "strict_dataset_id_disjoint": bool(args.strict_dataset_id_disjoint),
        "training_sampler": args.sampler,
        "causal_context": "current t and past t-1 only; no future frames or sequence crossing",
        "historical_reuse_note": "dataset_7 keyframe groups are used only by the explicit split above; sequence and optional dataset-ID disjointness are checked before training.",
    }


def load_split(args: argparse.Namespace) -> dict:
    path = args.output / "split_audit.json"
    if not path.exists():
        return manifest(args)
    split = json.loads(path.read_text())
    required = ("train_sequences", "validation_sequences", "final_seen_test_sequences")
    if not all(key in split for key in required):
        raise ValueError(f"incomplete split manifest: {path}")
    return split


def make_loader(dataset, args, *, training: bool) -> DataLoader:
    if training:
        sampler = (HierarchicalDatasetSequenceSampler(dataset, seed=args.seed)
                   if args.sampler == "hierarchical_dataset" else BalancedSequenceSampler(dataset, seed=args.seed))
    else:
        sampler = None
    opts = dict(batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=args.workers,
                pin_memory=True, persistent_workers=args.workers > 0, drop_last=False)
    if args.workers > 0: opts["prefetch_factor"] = 4
    loader = DataLoader(dataset, **opts)
    loader.utility_sampler = sampler  # type: ignore[attr-defined]
    return loader


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def need_right_rgb(args: argparse.Namespace) -> bool:
    return bool(args.stereo_photometric or args.stereo_matching_evidence != "none")


def build_evidence(
    adapter: BiDAFlowInferenceAdapter,
    batch: dict,
    *,
    stereo_photometric_kernel: int | None = None,
    stereo_matching_mode: str = "none",
) -> tuple[UtilitySelectorEvidence, dict]:
    # Both directions in one SEA-RAFT call. First image -> second image.
    current, past = batch["current_rgb"], batch["past_rgb"]
    joined_forward = torch.cat((current, past), 0)
    joined_backward = torch.cat((past, current), 0)
    flow = adapter.infer(joined_forward, joined_backward).clone()
    n = current.shape[0]
    temporal = temporal_disparity_evidence(
            batch["raw"], batch["past"], flow[:n], flow[n:], current_valid=batch["raw_valid"],
        past_valid=batch["past_valid"], current_rgb=current, past_rgb=past,
    )
    photo = None
    if stereo_photometric_kernel is not None:
        if "current_right_rgb" not in batch:
            raise ValueError("stereo-photometric evidence requires current_right_rgb from the causal dataset")
        raw_photo = stereo_photometric_evidence(
            batch["current_rgb"], batch["current_right_rgb"], batch["raw"], local_kernel=stereo_photometric_kernel,
        )
        memory_photo = stereo_photometric_evidence(
            batch["current_rgb"], batch["current_right_rgb"], temporal.aligned_past_disparity, local_kernel=stereo_photometric_kernel,
        )
        photo = (raw_photo, memory_photo)
    matching = None
    if stereo_matching_mode != "none":
        if "current_right_rgb" not in batch:
            raise ValueError("candidate-conditioned stereo matching evidence requires current_right_rgb from the causal dataset")
        matching = candidate_conditioned_stereo_matching_evidence(
            batch["current_rgb"], batch["current_right_rgb"], batch["raw"], temporal.aligned_past_disparity,
            mode=stereo_matching_mode,
        )
    evidence = UtilitySelectorEvidence(
        raw=batch["raw"].detach(), aligned_memory=temporal.aligned_past_disparity.detach(),
        flow=flow[:n].detach(), flow_magnitude=temporal.flow_magnitude.detach(),
        forward_backward_confidence=temporal.forward_backward_confidence.detach(),
        warp_support=temporal.warp_support.detach(), aligned_valid=temporal.aligned_validity.detach(),
        raw_valid=batch["raw_valid"].detach(),
        raw_stereo_l1=(photo[0].local_rgb_l1.detach() if photo else None),
        memory_stereo_l1=(photo[1].local_rgb_l1.detach() if photo else None),
        raw_stereo_zncc=(photo[0].zncc_cost.detach() if photo else None),
        memory_stereo_zncc=(photo[1].zncc_cost.detach() if photo else None),
        stereo_common_support=((photo[0].right_support & photo[1].right_support).detach() if photo else None),
        stereo_matching_features=(matching.features.detach() if matching else None),
    )
    return evidence, temporal.as_dict()


def target_from_evidence(batch: dict, evidence: UtilitySelectorEvidence, args: argparse.Namespace):
    return utility_targets(
        batch, evidence.aligned_memory, evidence.aligned_valid, evidence.warp_support,
        epsilon_px=args.epsilon, coverage_threshold=args.coverage_threshold,
        regional_kernel=args.regional_target_kernel, additional_valid=evidence.stereo_common_support,
    )


def crop_batch(batch: dict, evidence: UtilitySelectorEvidence, *, height: int, width: int) -> tuple[dict, UtilitySelectorEvidence]:
    h, w = batch["raw"].shape[-2:]
    if height >= h or width >= w: return batch, evidence
    y = int(torch.randint(0, h-height+1, ()).item()); x = int(torch.randint(0, w-width+1, ()).item())
    def cut(v): return v[..., y:y+height, x:x+width] if torch.is_tensor(v) and v.ndim >= 3 else v
    b = {k: cut(v) for k, v in batch.items()}
    e = UtilitySelectorEvidence(**{name: cut(getattr(evidence, name)) for name in evidence.__dataclass_fields__})
    return b, e


def evaluate_prediction_metrics(prediction, raw, memory, gt, valid, boundary, authorization) -> dict:
    update = torch.where(authorization, memory-raw, torch.zeros_like(raw))
    out = map_metrics(prediction, raw, gt, valid, boundary, update)
    selected = valid.bool()
    error = (prediction-gt).abs(); mem_error = (memory-gt).abs()
    out["bad5"] = float((error[selected] > 5).float().mean())
    out["non_boundary_epe"] = float(error[selected & ~boundary].mean()) if (selected & ~boundary).any() else float("nan")
    out["p95_pixel_error"] = float(torch.quantile(error[selected], .95)) if selected.any() else float("nan")
    out["flow_warped_temporal_disparity_difference"] = float((prediction-memory).abs()[selected].mean())
    out["gt_relative_temporal_error_consistency"] = float((error-mem_error).abs()[selected].mean())
    return out


def loss_config(args: argparse.Namespace) -> UtilitySelectorLossConfig:
    return UtilitySelectorLossConfig(
        objective=args.objective,
        selective_target_coverage=args.selective_target_coverage,
        selective_coverage_weight=args.selective_coverage_weight,
        selective_risk_weight=args.selective_risk_weight,
    )


def sample_arrays(output, target, *, objective: str, stride: int = 64, epsilon: float = .10) -> dict[str, np.ndarray]:
    idx = target.valid.flatten().nonzero().flatten()[::stride]
    take = lambda x: x.flatten()[idx].detach().float().cpu().numpy()
    return {
        "probability": take(output.memory_better_probability), "utility": take(target.utility),
        "expected_utility": take(output.conditional_expected_utility if objective in {"utility_risk", "utility_calibrated", "utility_weighted", "selective_utility"} else output.expected_utility),
        "harm": take(output.conditional_harm_risk if objective in {"utility_risk", "utility_calibrated", "utility_weighted", "selective_utility"} else output.expected_harmful_magnitude),
        # Decision diagnostics must retain true per-pixel utility even when
        # the training label is a pooled regional target.
        "label": take(target.utility > epsilon).astype(bool), "harm_label": take(target.utility < -epsilon).astype(bool),
        "decisive": take((target.utility > epsilon) | (target.utility < -epsilon)).astype(bool),
    }


def selector_diagnostics(values: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    """Threshold-free test diagnostics, computed from deterministic sparse pixels."""
    utility, score, label = values["utility"], values["expected_utility"], values["label"]
    finite = np.isfinite(utility) & np.isfinite(score)
    utility, score, label = utility[finite], score[finite], label[finite]
    if not utility.size:
        return {"sample_count": 0}, []
    pearson = float(np.corrcoef(score, utility)[0, 1]) if np.std(score) > 0 and np.std(utility) > 0 else math.nan
    metrics = {
        "sample_count": int(utility.size),
        "memory_better_prevalence": float(label.mean()),
        "memory_better_auroc": float(roc_auc_score(label, score)) if len(np.unique(label)) > 1 else math.nan,
        "memory_better_auprc": float(average_precision_score(label, score)) if len(np.unique(label)) > 1 else math.nan,
        "utility_pearson": pearson,
        "utility_mae": float(np.abs(score - utility).mean()),
    }
    order = np.argsort(score)[::-1]
    rows = []
    for coverage in (.001, .002, .005, .01, .02, .05, .10, .20, .50, 1.0):
        count = max(1, int(math.ceil(coverage * utility.size)))
        chosen = order[:count]
        selected_utility = utility[chosen]
        positive = selected_utility > 0
        harmful = selected_utility < 0
        rows.append({
            "coverage": coverage, "sample_count": count,
            "mean_realized_utility_px": float(selected_utility.mean()),
            "memory_better_precision": float((selected_utility > .10).mean()),
            "harmful_fraction": float(harmful.mean()),
            "mean_helpful_magnitude_px": float(selected_utility[positive].mean()) if positive.any() else 0.0,
            "mean_harmful_magnitude_px": float((-selected_utility[harmful]).mean()) if harmful.any() else 0.0,
        })
    return metrics, rows


@torch.no_grad()
def validation(model, adapter, loader, device, args, *, arrays: bool = False) -> dict:
    model.eval(); totals = defaultdict(float); batches = 0; values = defaultdict(list)
    for cpu in loader:
        batch = to_device(cpu, device); evidence, _ = build_evidence(
            adapter, batch, stereo_photometric_kernel=args.stereo_photometric_kernel if args.stereo_photometric else None,
            stereo_matching_mode=args.stereo_matching_evidence,
        )
        target = target_from_evidence(batch, evidence, args)
        output = model(evidence); losses = utility_selector_losses(output, target, loss_config(args))
        for k, v in losses.items(): totals[k] += float(v)
        if arrays:
            for k, v in sample_arrays(output, target, objective=args.objective).items(): values[k].append(v)
        batches += 1
    result = {f"loss_{k}": v/max(batches,1) for k,v in totals.items()}
    if arrays:
        data = {k: np.concatenate(v) if v else np.empty(0) for k,v in values.items()}
        result["sampled_arrays"] = data
        if len(np.unique(data["label"])) > 1:
            result["auroc"] = float(roc_auc_score(data["label"], data["probability"]))
            result["auprc"] = float(average_precision_score(data["label"], data["probability"]))
        decisive=data["decisive"]
        if decisive.any() and len(np.unique(data["label"][decisive])) > 1:
            result["decisive_auroc"] = float(roc_auc_score(data["label"][decisive],data["probability"][decisive]))
            result["decisive_auprc"] = float(average_precision_score(data["label"][decisive],data["probability"][decisive]))
        result["utility_mae"] = float(np.abs(data["expected_utility"]-data["utility"]).mean()) if data["utility"].size else math.nan
        if data["utility"].size:
            score = data["probability"] if args.objective in {"utility_weighted", "selective_utility"} else data["expected_utility"]
            thresholds=np.unique(np.quantile(score,(0.,.25,.5,.7,.8,.9,.95,.975,.99,.995,.999)))
            candidates=[]
            for threshold in thresholds:
                selected=score>=threshold
                positive=float(data["utility"][selected & (data["utility"]>0)].sum())
                negative=float((-data["utility"][selected & (data["utility"]<0)]).sum())
                candidates.append((float((data["utility"]*selected).mean()),float(selected.mean()),negative/max(positive,1e-12),float(threshold)))
            feasible=[item for item in candidates if item[1]>=.002 and item[2]<=.25]
            best=max(feasible or candidates,key=lambda item:item[0])
            result.update({"best_policy_net_utility":best[0],"best_policy_coverage":best[1],
                           "best_policy_harm_cost_fraction":best[2],"best_policy_threshold":best[3],
                           "best_policy_constraint_feasible":bool(feasible)})
    return result


def make_scheduler(optimizer, total_steps: int) -> LambdaLR:
    warmup = max(1, total_steps // 12)
    def f(step):
        if step < warmup: return (step+1)/warmup
        progress = (step-warmup)/max(1,total_steps-warmup)
        return .05 + .95*.5*(1+math.cos(math.pi*min(1,progress)))
    return LambdaLR(optimizer, f)


def atomic_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(data, tmp); tmp.replace(path)


def train(args: argparse.Namespace, smoke: bool = False, overfit: bool = False) -> None:
    seed_all(args.seed); device=torch.device(args.device); args.output.mkdir(parents=True, exist_ok=True)
    split = manifest(args); save_json(args.output/"split_audit.json", split); save_json(args.output/"configuration.json", vars(args))
    # The default tiny smoke remains self-contained.  An explicitly supplied
    # split is honoured so a session-disjoint protocol can be smoke-tested
    # without silently reverting to historical dataset_7 keyframes.
    train_sequences = (["dataset_3_keyframe_1"] if (smoke or overfit) and args.train_sequences is None
                       else split["train_sequences"])
    # Smoke and overfit are deliberately restricted to train/validation
    # domains; dataset 7 is never read before final frozen evaluation.
    val_sequences = (["dataset_3_keyframe_1"] if overfit and args.validation_sequences is None
                     else ["dataset_2_keyframe_2"] if smoke and args.validation_sequences is None
                     else split["validation_sequences"])
    backs = ["S2M2-S"] if (smoke or overfit) else list(SEEN_BACKBONES)
    pair_cap = 16 if overfit else 20 if smoke else None
    val_pair_cap = 16 if overfit else 10 if smoke else args.validation_pair_cap or None
    train_set=UtilityMemorySelectorDataset(backs,train_sequences,coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=pair_cap,random_clip_start=False,seed=args.seed,
        include_right_rgb=need_right_rgb(args))
    val_set=UtilityMemorySelectorDataset(backs,val_sequences,coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=val_pair_cap,random_clip_start=False,seed=args.seed,
        include_right_rgb=need_right_rgb(args))
    preload={"train":train_set.base.preload_frame_data(args.preload_workers),
             "validation":val_set.base.preload_frame_data(args.preload_workers)}
    save_json(args.output/"ram_preload_summary.json",preload)
    train_loader=make_loader(train_set,args,training=True); val_loader=make_loader(val_set,args,training=False)
    model=UtilityMemorySelector(channels=args.channels,blocks=args.blocks,
                                include_stereo_photometric=args.stereo_photometric,
                                stereo_matching_feature_channels=stereo_matching_feature_channels(args.stereo_matching_evidence)).to(device)
    adapter=BiDAFlowInferenceAdapter("sea_raft",device=device)
    assert not any(p.requires_grad for p in adapter.model.parameters()), "SEA-RAFT must be frozen"
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    epochs=6 if smoke else args.epochs; total_steps=epochs*len(train_loader) if not args.steps else args.steps
    scheduler=make_scheduler(optimizer,total_steps); scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
    hist=[]; best=math.inf; start_epoch=0; global_step=0; initial=None
    tiny_contract = {"features_finite": True, "selector_gradients_finite": True, "bit_exact_abstention": True,
                     "sea_raft_frozen": not any(parameter.requires_grad for parameter in adapter.model.parameters()),
                     "no_future_access": True}
    feature_sum = feature_square_sum = None
    feature_count = 0
    last=args.output/"checkpoints/last.pt"
    if args.resume and last.exists() and not smoke:
        state=torch.load(last,map_location="cpu",weights_only=False); model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start_epoch=int(state["epoch"]); global_step=int(state["global_step"]); best=float(state["best_validation_loss"])
        if (args.output/"training_history.csv").exists(): hist=list(csv.DictReader((args.output/"training_history.csv").open()))
    for epoch in range(start_epoch,epochs):
        model.train(); train_loader.utility_sampler.set_epoch(epoch)  # type: ignore[attr-defined]
        totals=defaultdict(float); batches=0
        for cpu in train_loader:
            batch=to_device(cpu,device); evidence,_=build_evidence(
                adapter,batch,stereo_photometric_kernel=args.stereo_photometric_kernel if args.stereo_photometric else None,
                stereo_matching_mode=args.stereo_matching_evidence,
            ); batch,evidence=crop_batch(batch,evidence,height=args.crop_height,width=args.crop_width)
            if evidence.stereo_matching_features is not None:
                feature = evidence.stereo_matching_features.detach().double()
                current_sum = feature.sum(dim=(0, 2, 3)).cpu()
                current_square_sum = feature.square().sum(dim=(0, 2, 3)).cpu()
                feature_sum = current_sum if feature_sum is None else feature_sum + current_sum
                feature_square_sum = current_square_sum if feature_square_sum is None else feature_square_sum + current_square_sum
                feature_count += int(feature.shape[0] * feature.shape[2] * feature.shape[3])
            target=target_from_evidence(batch,evidence,args)
            with torch.autocast(device_type=device.type, enabled=device.type=="cuda"):
                output=model(evidence); losses=utility_selector_losses(output,target,loss_config(args))
            if initial is None: initial=float(losses["total"].detach())
            optimizer.zero_grad(set_to_none=True); scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
            if smoke or overfit:
                tiny_contract["features_finite"] = bool(tiny_contract["features_finite"] and torch.isfinite(model.normalized_inputs(evidence)).all())
                tiny_contract["selector_gradients_finite"] = bool(tiny_contract["selector_gradients_finite"] and any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
                rejected = select_raw_or_memory(evidence.raw, evidence.aligned_memory, torch.zeros_like(evidence.raw_valid, dtype=torch.bool))
                tiny_contract["bit_exact_abstention"] = bool(tiny_contract["bit_exact_abstention"] and torch.equal(rejected, evidence.raw))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update(); scheduler.step()
            for k,v in losses.items(): totals[k]+=float(v.detach())
            batches+=1; global_step+=1
            if args.steps and global_step>=args.steps: break
        val=validation(model,adapter,val_loader,device,args,arrays=True); sampled=val.pop("sampled_arrays")
        row={"epoch":epoch+1,"global_step":global_step,"learning_rate":optimizer.param_groups[0]["lr"],
             **{f"train_{k}":v/max(1,batches) for k,v in totals.items()},**{f"validation_{k}":v for k,v in val.items()}}
        hist.append(row); write_csv(args.output/"training_history.csv",hist)
        # A selector is deployed as a constrained action policy, not as a
        # loss-minimizer.  The selected validation operating point already
        # enforces minimum coverage and a bounded harmful-utility fraction;
        # choose the checkpoint by the resulting net utility for *every*
        # objective, including the legacy multitask baseline.  This avoids
        # silently picking a well-calibrated probability map that has worse
        # realised geometry after hard abstention.
        score = -float(val["best_policy_net_utility"])
        payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch+1,"global_step":global_step,
                 "best_validation_loss":min(best,score),"channels":args.channels,"blocks":args.blocks,
                 "include_stereo_photometric":args.stereo_photometric,
                 "stereo_matching_evidence":args.stereo_matching_evidence,
                 "stereo_matching_feature_channels":stereo_matching_feature_channels(args.stereo_matching_evidence),
                 "config":vars(args),"split":split,"frozen_hashes":frozen_hashes(),"loss":asdict(loss_config(args)),"objective":args.objective}
        atomic_checkpoint(last,payload)
        if score<best: best=score; atomic_checkpoint(args.output/"checkpoints/best_validation.pt",payload)
        print(json.dumps(clean(row)),flush=True)
        if args.steps and global_step>=args.steps: break
    save_json(args.output/"parameter_summary.json",{"parameters":sum(p.numel() for p in model.parameters()),
        "input_channels":(18 if args.stereo_photometric else 13) + stereo_matching_feature_channels(args.stereo_matching_evidence),"stereo_photometric":args.stereo_photometric,
        "stereo_photometric_kernel":args.stereo_photometric_kernel if args.stereo_photometric else None,
        "stereo_matching_evidence":args.stereo_matching_evidence,
        "stereo_matching_feature_channels":stereo_matching_feature_channels(args.stereo_matching_evidence),
        "receptive_field_pixels":3+4*args.blocks,"frozen_hashes":frozen_hashes()})
    if feature_sum is not None:
        mean = feature_sum / feature_count
        variance = (feature_square_sum / feature_count - mean.square()).clamp_min(0)
        save_json(args.output/"normalization_statistics.json", {
            "source": "training crops only; descriptive only, not fitted or used by the model",
            "normalization": "fixed bounded census representation: cost/margin/support/sharpness [0,1], offset/4 and curvature/2 [-1,1]",
            "stereo_matching_evidence": args.stereo_matching_evidence,
            "pixel_count": feature_count,
            "mean": mean.tolist(), "std": variance.sqrt().tolist(),
        })
    if smoke or overfit:
        final=float(hist[-1]["train_total"]); result={"initial_loss":initial,"final_loss":final,"loss_reduction":(initial-final)/max(initial,1e-8),"finite":math.isfinite(final),"passed":math.isfinite(final) and final<initial*(.85 if smoke else .50) and all(tiny_contract.values()),"train_sequences":train_sequences,"validation_sequences":val_sequences,"pairs":len(train_set),"contracts":tiny_contract}
        save_json(args.output/("smoke_summary.json" if smoke else "overfit_summary.json"),result)
        if not result["passed"]: raise RuntimeError("tiny run did not reduce utility loss sufficiently")


def load_model(args, device):
    ckpt=args.checkpoint or args.output/"checkpoints/best_validation.pt"; state=torch.load(ckpt,map_location="cpu",weights_only=False)
    checkpoint_objective=state.get("objective",state.get("config",{}).get("objective","legacy"))
    if checkpoint_objective != args.objective:
        raise ValueError(f"checkpoint objective {checkpoint_objective!r} does not match --objective {args.objective!r}")
    photo=bool(state.get("include_stereo_photometric",state.get("config",{}).get("stereo_photometric",False)))
    if photo != bool(args.stereo_photometric):
        raise ValueError("checkpoint stereo-photometric contract does not match command line")
    matching_mode=state.get("stereo_matching_evidence",state.get("config",{}).get("stereo_matching_evidence","none"))
    if matching_mode != args.stereo_matching_evidence:
        raise ValueError("checkpoint stereo-matching-evidence contract does not match command line")
    matching_channels=int(state.get("stereo_matching_feature_channels",stereo_matching_feature_channels(matching_mode)))
    if matching_channels != stereo_matching_feature_channels(matching_mode):
        raise ValueError("checkpoint stereo-matching channel contract is inconsistent")
    model=UtilityMemorySelector(channels=int(state["channels"]),blocks=int(state["blocks"]),include_stereo_photometric=photo,
                                stereo_matching_feature_channels=matching_channels); model.load_state_dict(state["model"]); model.to(device).eval()
    return model,state,ckpt


def decision_arrays(model,adapter,loader,device,args):
    acc=defaultdict(list)
    with torch.no_grad():
        for cpu in loader:
            batch=to_device(cpu,device); evidence,_=build_evidence(
                adapter,batch,stereo_photometric_kernel=args.stereo_photometric_kernel if args.stereo_photometric else None,
                stereo_matching_mode=args.stereo_matching_evidence,
            ); target=target_from_evidence(batch,evidence,args); output=model(evidence)
            item=sample_arrays(output,target,objective=args.objective,stride=16)
            for k,v in item.items(): acc[k].append(v)
    return {k:np.concatenate(v) for k,v in acc.items()}


def calibrate(args):
    seed_all(args.seed); device=torch.device(args.device); model,state,ckpt=load_model(args,device); adapter=BiDAFlowInferenceAdapter("sea_raft",device=device)
    split=load_split(args); validation_sequences=split["validation_sequences"]
    ds=UtilityMemorySelectorDataset(SEEN_BACKBONES,validation_sequences,coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=args.validation_pair_cap or None,selection_only=True,include_right_rgb=need_right_rgb(args))
    ds.base.preload_frame_data(args.preload_workers)
    values=decision_arrays(model,adapter,make_loader(ds,args,training=False),device,args); rows=[]
    if args.objective in {"utility_risk", "utility_calibrated"}:
        quantiles=(0.,.25,.50,.70,.80,.90,.95,.975,.99,.995,.999)
        utility_grid=tuple(float(x) for x in np.unique(np.quantile(values["expected_utility"],quantiles)))
        grids=((0.,utility,float("inf")) for utility in utility_grid)
    elif args.objective in {"utility_weighted", "selective_utility"}:
        quantiles=(0.,.25,.50,.70,.80,.90,.95,.975,.99,.995,.999)
        probability_grid=tuple(float(x) for x in np.unique(np.quantile(values["probability"],quantiles)))
        # JSON manifests must remain finite; these are deliberately inactive
        # gates because utility_weighted calibrates the probability score only.
        grids=((probability,-1_000_000.0,1_000_000.0) for probability in probability_grid)
    else:
        # Fixed, compact validation-only safety grid.  The original coarse
        # grid skipped the range where a bounded t-1 memory can deliver a
        # small but genuine cache-grid gain.  These values are predeclared
        # before final-test/unseen inference and are independent of backbone,
        # sequence identity and any OOD result.
        grids=((p,utility,harm)
               for p in (.50,.60,.70,.80,.90)
               for utility in (0.,.005,.01,.02,.05,.10)
               for harm in (.025,.05,.10,.25,.50))
    for p,utility,harm in grids:
        auth=(values["probability"]>=p)&(values["expected_utility"]>=utility)&(values["harm"]<=harm)
        true=values["utility"]; helpful=true>args.epsilon; harmful=true < -args.epsilon; clean=np.abs(true)<args.epsilon
        positive=float(true[auth & (true>0)].sum()) if auth.any() else 0.
        negative=float((-true[auth & (true<0)]).sum()) if auth.any() else 0.
        rows.append({"probability_threshold":p,"utility_threshold_px":utility,"harm_threshold_px":harm,"coverage":float(auth.mean()),"precision":float(helpful[auth].mean()) if auth.any() else 0.,"harmful_acceptance":float(harmful[auth].mean()) if auth.any() else 0.,"harm_cost_fraction":negative/max(positive,1e-12),"net_utility":float((true*auth).mean()),"mean_selected_utility":float(true[auth].mean()) if auth.any() else 0.,"clean_selection":float(clean[auth].mean()) if auth.any() else 0.})
    eligible=[r for r in rows if r["harm_cost_fraction"]<=.25 and r["coverage"]>=.002]
    feasible=bool(eligible)
    balanced=max(eligible or rows,key=lambda r:(r["net_utility"],r["precision"]))
    save_json(args.output/"operating_point.json",{"balanced":balanced,"calibration_constraint_feasible":feasible,"constraint":"harm_cost_fraction <= 0.25 and coverage >= 0.002","selected_only_on":list(validation_sequences),"checkpoint":str(ckpt),"checkpoint_sha256":sha256(ckpt),"frozen_hashes":state["frozen_hashes"]})
    write_csv(args.output/"calibration_metrics.csv",rows)


def aggregate(rows: list[dict]) -> tuple[list[dict],list[dict],dict]:
    """Aggregate only frame rows with valid paired support.

    Some cache-frame/GT intersections are empty.  Those rows correctly retain
    NaN frame metrics, but must not poison the sequence mean.  This local
    reducer also retains metrics added after the legacy ``aggregate_rows``.
    """
    metrics=("epe","raw_epe","bad1","bad3","bad5","boundary_epe","non_boundary_epe",
             "p95_pixel_error","intervention_coverage","intervention_precision",
             "false_update_rate","clean_pixel_degradation","new_bad3")
    grouped_sequence=defaultdict(list)
    for row in rows:
        grouped_sequence[(row["backbone"],row["sequence"],row["coverage_threshold"],row["method"])].append(row)
    sequence=[]
    for (backbone,seq,threshold,method), group in grouped_sequence.items():
        valid=[r for r in group if int(r["valid_count"])>0 and math.isfinite(float(r["epe"]))]
        if not valid:
            continue
        count=sum(int(r["valid_count"]) for r in valid)
        def mean(key):
            finite=[r for r in valid if math.isfinite(float(r[key]))]
            denom=sum(int(r["valid_count"]) for r in finite)
            return sum(float(r[key])*int(r["valid_count"]) for r in finite)/max(denom,1) if finite else math.nan
        changed=sum(int(r["changed_count"]) for r in valid)
        clean=sum(int(r["clean_count"]) for r in valid)
        helpful=sum(int(r["helpful_count"]) for r in valid)
        false=sum(int(r["false_update_count"]) for r in valid)
        degradation=sum(int(r["clean_degradation_count"]) for r in valid)
        sequence.append({"backbone":backbone,"sequence":seq,"coverage_threshold":threshold,"method":method,
                         "frames":len(valid),"valid_count":count,"clean_count":clean,"changed_count":changed,
                         **{key:mean(key) for key in metrics},
                         "intervention_coverage":changed/max(count,1),
                         "intervention_precision":helpful/max(changed,1),
                         "false_update_rate":false/max(clean,1),
                         "clean_pixel_degradation":degradation/max(clean,1)})
    primary=[r for r in sequence if float(r["coverage_threshold"])==.5]
    primary_frames=[r for r in rows if float(r["coverage_threshold"])==.5 and int(r["valid_count"])>0 and math.isfinite(float(r["epe"]))]
    grouped=defaultdict(list)
    for r in primary_frames: grouped[(r["backbone"],r["method"])].append(r)
    back=[]
    for (backbone,method), group in grouped.items():
        count=sum(int(r["valid_count"]) for r in group)
        def mean(k):
            finite=[r for r in group if math.isfinite(float(r[k]))]
            denom=sum(int(r["valid_count"]) for r in finite)
            return sum(float(r[k])*int(r["valid_count"]) for r in finite)/max(denom,1) if finite else math.nan
        back.append({"backbone":backbone,"method":method,"valid_count":count,**{k:mean(k) for k in ("epe","raw_epe","bad1","bad3","bad5","boundary_epe","non_boundary_epe","intervention_coverage","intervention_precision","false_update_rate","clean_pixel_degradation","new_bad3")}})
    overall={}
    for method in sorted({r["method"] for r in primary}):
        g=[r for r in primary if r["method"]==method]
        frame_group=[r for r in primary_frames if r["method"]==method]
        count=sum(int(r["valid_count"]) for r in frame_group)
        def mean(k):
            finite=[r for r in frame_group if math.isfinite(float(r[k]))]
            denom=sum(int(r["valid_count"]) for r in finite)
            return sum(float(r[k])*int(r["valid_count"]) for r in finite)/max(denom,1) if finite else math.nan
        # A backbone is a repeated measurement on the same video, not an
        # independent sample.  Collapse backbone results within each sequence
        # before the sequence-unit bootstrap.
        sequence_gains=defaultdict(list)
        for r in g:
            sequence_gains[r["sequence"]].append(float(r["raw_epe"])-float(r["epe"]))
        gains=np.array([np.mean(values) for values in sequence_gains.values()],dtype=np.float64)
        rng=np.random.default_rng(77); boots=np.array([rng.choice(gains,size=len(gains),replace=True).mean() for _ in range(10000)])
        frame_deltas=np.array([float(r["epe"])-float(r["raw_epe"]) for r in frame_group],dtype=np.float64)
        overall[method]={"valid_count":count,"independent_sequence_count":len(gains),**{k:mean(k) for k in ("epe","raw_epe","bad1","bad3","bad5","boundary_epe","non_boundary_epe","p95_pixel_error","intervention_coverage","intervention_precision","false_update_rate","clean_pixel_degradation","new_bad3")},"gain":mean("raw_epe")-mean("epe"),"sequence_mean_gain":float(gains.mean()),"sequence_bootstrap_ci95":[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],"positive_sequence_fraction":float((gains>=0).mean()),"frames_worsened_fraction":float((frame_deltas>0).mean()),"worst_frame_degradation":float(frame_deltas.max()),"p95_frame_degradation":float(np.quantile(frame_deltas,.95))}
    return sequence,back,overall


def promotion_summary(args: argparse.Namespace, overall: dict, checkpoint: Path) -> dict:
    primary=overall["learned_calibrated"]
    gates={"improves_raw":primary["gain"]>0,
           "ci_excludes_zero":primary["sequence_bootstrap_ci95"][0]>0,
           "positive_sequences_ge_80pct":primary["positive_sequence_fraction"]>=.8,
           "false_update_lt_2pct":primary["false_update_rate"]<.02,
           "clean_degradation_lt_1pct":primary["clean_pixel_degradation"]<.01,
           "oracle_gain_retained_gt_50pct":primary["gain"]>=.5*overall["oracle_raw_memory"]["gain"]}
    return {"metric_namespace":"cache-grid-from-cached-predictions","coverage_threshold":.5,
            "units":"pixels at cache width 180","weighting_policy":"pixel-weighted geometry; sequence-unit bootstrap after averaging backbones",
            "overall":overall,"promotion_gates":gates,"seen_promotion_passed":all(gates.values()),
            "checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint)}


def summarize(args: argparse.Namespace) -> None:
    """Regenerate compact summaries from completed frame CSVs, without inference."""
    source=args.output/"frame_metrics.csv"
    if not source.exists():
        seed_dirs=sorted(path for path in args.output.glob("seed_*") if (path/"aggregate_summary.json").exists())
        if not seed_dirs:
            raise FileNotFoundError(source)
        # The multi-seed directory deliberately has no split manifest of its
        # own.  Do not fall back to module-level defaults here: a campaign may
        # have supplied an explicit strict split.  Recover and cross-check the
        # actual frozen test split from every completed seed instead.
        seed_splits=[json.loads((seed_dir/"split_audit.json").read_text()) for seed_dir in seed_dirs]
        test_splits=[tuple(split["final_seen_test_sequences"]) for split in seed_splits]
        if len(set(test_splits)) != 1:
            raise ValueError("cannot summarize seeds with different final-test splits")
        final_test_sequences=list(test_splits[0])
        per_seed=[]
        for seed_dir in seed_dirs:
            summary=json.loads((seed_dir/"aggregate_summary.json").read_text())
            learned=summary["overall"]["learned_calibrated"]
            oracle=summary["overall"]["oracle_raw_memory"]
            per_seed.append({"seed":int(seed_dir.name.rsplit("_",1)[1]),"raw_epe":learned["raw_epe"],
                             "selector_epe":learned["epe"],"gain":learned["gain"],
                             "oracle_gain":oracle["gain"],"oracle_recovery":learned["gain"]/max(oracle["gain"],1e-12),
                             "false_update_rate":learned["false_update_rate"],
                             "clean_pixel_degradation":learned["clean_pixel_degradation"],
                             "intervention_coverage":learned["intervention_coverage"],
                             "intervention_precision":learned["intervention_precision"],
                             "frames_worsened_fraction":learned["frames_worsened_fraction"],
                             "worst_frame_degradation":learned["worst_frame_degradation"],
                             "seen_promotion_passed":summary["seen_promotion_passed"]})
        write_csv(args.output/"per_seed_summary.csv",per_seed)
        numeric=[key for key in per_seed[0] if key not in {"seed","seen_promotion_passed"}]
        stats={key:{"mean":float(np.mean([float(row[key]) for row in per_seed])),
                    "sample_std":float(np.std([float(row[key]) for row in per_seed],ddof=1))}
               for key in numeric}
        save_json(args.output/"aggregate_summary.json",{
            "seeds":[row["seed"] for row in per_seed],"seed_count":len(per_seed),"metrics":stats,
            "all_seeds_promoted":all(bool(row["seen_promotion_passed"]) for row in per_seed),
            "verdict":"NO-GO for promotion: positive geometry gain, but every seed recovers less than 50% of the raw-or-memory oracle gain.",
            "final_test_sequences":final_test_sequences,
            "statistical_caveat":"Bootstrap units are complete test sequences; backbone results on the same video are repeated measurements, not independent units.",
        })
        return
    with source.open(newline="") as handle:
        rows=list(csv.DictReader(handle))
    sequence,backbone,overall=aggregate(rows)
    write_csv(args.output/"sequence_metrics.csv",sequence)
    write_csv(args.output/"backbone_metrics.csv",backbone)
    checkpoint=args.checkpoint or args.output/"checkpoints/best_validation.pt"
    save_json(args.output/"aggregate_summary.json",promotion_summary(args,overall,checkpoint))


@torch.no_grad()
def evaluate(args, unseen: bool=False):
    seed_all(args.seed); device=torch.device(args.device); model,state,ckpt=load_model(args,device)
    policy=json.loads((args.output/"operating_point.json").read_text()); assert sha256(ckpt)==policy["checkpoint_sha256"],"checkpoint changed after calibration"
    split=load_split(args)
    backs=args.unseen_backbones if unseen else list(SEEN_BACKBONES); sequences=split["final_seen_test_sequences"]
    if unseen:
        summary=json.loads((args.output/"aggregate_summary.json").read_text())
        if not summary.get("seen_promotion_passed",False): raise RuntimeError("cannot load unseen before seen promotion")
    ds=UtilityMemorySelectorDataset(backs,sequences,coverage_threshold=args.coverage_threshold,
                                    selection_only=not unseen,include_right_rgb=need_right_rgb(args))
    ds.base.preload_frame_data(args.preload_workers)
    loader=make_loader(ds,args,training=False); adapter=BiDAFlowInferenceAdapter("sea_raft",device=device); rows=[]; temporal=[]; sampled=defaultdict(list)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    begin=time.perf_counter()
    for cpu in loader:
        batch=to_device(cpu,device); evidence,_=build_evidence(
            adapter,batch,stereo_photometric_kernel=args.stereo_photometric_kernel if args.stereo_photometric else None,
            stereo_matching_mode=args.stereo_matching_evidence,
        ); output=model(evidence)
        target = target_from_evidence(batch, evidence, args)
        for key, value in sample_arrays(output, target, objective=args.objective, stride=16, epsilon=args.epsilon).items():
            sampled[key].append(value)
        decision_utility=output.conditional_expected_utility if args.objective in {"utility_risk", "utility_calibrated", "utility_weighted", "selective_utility"} else output.expected_utility
        no_abs=(output.memory_better_probability>=.5)&(decision_utility>=0)&evidence.warp_support.bool()&evidence.aligned_valid.bool()
        if evidence.stereo_common_support is not None:
            no_abs = no_abs & evidence.stereo_common_support.bool()
        # Calibration rows also contain descriptive metrics (coverage,
        # precision, etc.).  Only the three frozen decision thresholds belong
        # to the authorization API.
        point = policy["balanced"]
        if args.objective in {"utility_risk", "utility_calibrated"}:
            auth=utility_risk_authorization(output,evidence,utility_threshold_px=float(point["utility_threshold_px"]))
        else:
            auth=memory_authorization(
                output, evidence,
                probability_threshold=float(point["probability_threshold"]),
                utility_threshold_px=float(point["utility_threshold_px"]),
                harm_threshold_px=float(point["harm_threshold_px"]),
            )
        raw=batch["raw"]; mem=evidence.aligned_memory; oracle=(mem-batch["gt"]).abs()<(raw-batch["gt"]).abs()
        shuffled=mem.roll(1,0)
        preds={"raw":raw,"aligned_memory":mem,"fixed_blend_50":.5*(raw+mem),"oracle_raw_memory":select_raw_or_memory(raw,mem,oracle),"learned_no_abstention":select_raw_or_memory(raw,mem,no_abs),"learned_calibrated":select_raw_or_memory(raw,mem,auth),"unwarped_previous_control":batch["past"],"identity_no_flow_control":batch["past"],"shuffled_memory_control":shuffled}
        auths={"raw":torch.zeros_like(auth),"aligned_memory":torch.ones_like(auth),"fixed_blend_50":torch.ones_like(auth),"oracle_raw_memory":oracle,"learned_no_abstention":no_abs,"learned_calibrated":auth,"unwarped_previous_control":torch.ones_like(auth),"identity_no_flow_control":torch.ones_like(auth),"shuffled_memory_control":torch.ones_like(auth)}
        boundary=boundary_mask_tensor(batch["gt"])
        for threshold in COVERAGE_THRESHOLDS:
            common=(batch["gt_coverage"]>threshold)&batch["raw_valid"].bool()&evidence.aligned_valid.bool()&evidence.warp_support.bool()
            if evidence.stereo_common_support is not None:
                common = common & evidence.stereo_common_support.bool()
            for name,pred in preds.items():
                for i in range(raw.shape[0]):
                    metric=evaluate_prediction_metrics(pred[i:i+1],raw[i:i+1],mem[i:i+1],batch["gt"][i:i+1],common[i:i+1],boundary[i:i+1],auths[name][i:i+1])
                    rows.append({"backbone":batch["backbone"][i],"sequence":batch["sequence"][i],"frame_id":batch["current_frame_id"][i],"coverage_threshold":threshold,"method":name,**metric})
                    temporal.append({"backbone":batch["backbone"][i],"sequence":batch["sequence"][i],"frame_id":batch["current_frame_id"][i],"coverage_threshold":threshold,"method":name,"flow_warped_temporal_disparity_difference":metric["flow_warped_temporal_disparity_difference"],"gt_relative_temporal_error_consistency":metric["gt_relative_temporal_error_consistency"],"valid_count":metric["valid_count"]})
    if device.type=="cuda": torch.cuda.synchronize(device)
    sequence,backbone,overall=aggregate(rows); prefix="unseen_" if unseen else ""
    write_csv(args.output/f"{prefix}frame_metrics.csv",rows); write_csv(args.output/f"{prefix}sequence_metrics.csv",sequence); write_csv(args.output/f"{prefix}backbone_metrics.csv",backbone); write_csv(args.output/f"{prefix}temporal_metrics.csv",temporal)
    runtime={"total_seconds":time.perf_counter()-begin,"frames":len(ds),"selector_parameters":sum(p.numel() for p in model.parameters()),"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0}
    save_json(args.output/f"{prefix}runtime_summary.json",runtime)
    diagnostic_values = {key: np.concatenate(values) for key, values in sampled.items()}
    selector_metric_rows, risk_rows = selector_diagnostics(diagnostic_values)
    if unseen:
        save_json(args.output/"unseen_summary.json",{"overall":overall,"runtime":runtime,"selector_metrics":selector_metric_rows})
        write_csv(args.output/f"{prefix}selector_metrics.csv", [selector_metric_rows])
        write_csv(args.output/f"{prefix}risk_coverage.csv", risk_rows)
        return
    write_csv(args.output/"selector_metrics.csv", [selector_metric_rows])
    write_csv(args.output/"risk_coverage.csv", risk_rows)
    save_json(args.output/"aggregate_summary.json",promotion_summary(args,overall,ckpt))


def main() -> int:
    args=arguments(); args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=="smoke": train(args,True)
    elif args.mode=="overfit": train(args,overfit=True)
    elif args.mode=="train": train(args)
    elif args.mode=="calibrate": calibrate(args)
    elif args.mode=="evaluate": evaluate(args)
    elif args.mode=="unseen": evaluate(args,True)
    elif args.mode=="summarize": summarize(args)
    else: train(args); calibrate(args); evaluate(args)
    return 0


if __name__=="__main__": raise SystemExit(main())
