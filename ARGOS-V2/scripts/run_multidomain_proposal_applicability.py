#!/usr/bin/env python3
"""Train only frozen-A2 proposal applicability on SCARED-C plus D4D anchors.

This is deliberately a narrow control.  It reuses the validated P4 model and
loss unchanged; the only difference from the original P4 study is balanced
supervised exposure to a second domain whose geometry cache was audited.  No
unseen backbone, SERV-CT, StereoMIS or D4D specimen_3 is constructed here.
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from argos_v2.sequences import accepted_sequences  # noqa: E402
from model_design.data.multidomain_raw_error_dataset import (  # noqa: E402
    D4DAnchorDataset, DomainBalancedSampler, MultiDomainRawErrorDataset,
)
from model_design.data.proposal_utility_dataset import (  # noqa: E402
    ProposalUtilityDataset, proposal_utility_targets, stratified_training_targets,
)
from model_design.data.raw_error_dataset import CALIBRATION_SEQUENCES, TEST_SEQUENCES  # noqa: E402
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter, SEA_RAFT_CHECKPOINT  # noqa: E402
from model_design.losses.proposal_utility_losses import proposal_utility_losses  # noqa: E402
from model_design.models.proposal_applicability_detector import (  # noqa: E402
    ProposalApplicabilityDetector, apply_frozen_proposal, proposal_authorization_mask,
)
from run_learned_t1_refiner import build_evidence  # noqa: E402
from run_proposal_applicability import (  # noqa: E402
    binary_metrics, correlation, frozen_proposal_evidence, loss_config,
)
from run_raw_error_abstention import boundary_mask_tensor, load_a2, map_metrics, to_device  # noqa: E402


SEEN = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "train", "calibrate", "final", "finalize"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a2-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--d4d-fraction", type=float, choices=(.25, .50), default=.50)
    parser.add_argument("--d4d-train-specimens", default="specimen_1",
                        help="comma-separated supervised D4D specimens; never includes final specimen")
    parser.add_argument("--d4d-calibration-specimens", default="specimen_2",
                        help="comma-separated disjoint D4D calibration specimens, or empty with --scared-only-calibration")
    parser.add_argument("--scared-only-calibration", action="store_true",
                        help="exploratory cross-specimen control: select checkpoint/policy only on SCARED validation")
    parser.add_argument("--max-train-pairs", type=int, default=256)
    parser.add_argument("--max-validation-pairs", type=int, default=160)
    parser.add_argument("--training-pixels-per-batch", type=int, default=32768)
    parser.add_argument("--epsilon", type=float, default=.10)
    parser.add_argument("--coverage-threshold", type=float, default=.50)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save_json(path: Path, value: object) -> None:
    def clean(item):
        if isinstance(item, dict): return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)): return [clean(value) for value in item]
        if isinstance(item, np.generic): return clean(item.item())
        if isinstance(item, float) and not math.isfinite(item): return None
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    names: list[str] = []
    for row in rows:
        names.extend(key for key in row if key not in names)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader(); writer.writerows(rows)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def split() -> dict:
    held = set(CALIBRATION_SEQUENCES) | set(TEST_SEQUENCES)
    return {
        "train_sequences": [item for item in accepted_sequences() if item not in held],
        "calibration_sequences": list(CALIBRATION_SEQUENCES),
        "final_seen_sequences": list(TEST_SEQUENCES),
        "scared_backbones": list(SEEN),
        "d4d_train": ["specimen_1"], "d4d_calibration": ["specimen_2"],
        "d4d_final_only": ["specimen_3"], "d4d_backbones": ["S2M2-S"],
        "forbidden_before_freeze": ["D4D/specimen_3", "SERV-CT", "StereoMIS", "Fast-FoundationStereo", "CREStereo"],
        "causal_context": "t-1 -> t only; SEA-RAFT/BiDA/A2 frozen",
    }


def specimen_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"specimen_1", "specimen_2", "specimen_3"}
    if not set(result).issubset(allowed) or len(set(result)) != len(result):
        raise ValueError(f"invalid D4D specimen list: {value!r}")
    return result


def sources(config: argparse.Namespace, *, smoke: bool):
    manifest = split()
    train_sequences = [manifest["train_sequences"][0]] if smoke else manifest["train_sequences"]
    train_backbones = ["S2M2-S"] if smoke else list(SEEN)
    scared_train = ProposalUtilityDataset(train_backbones, train_sequences,
        coverage_threshold=config.coverage_threshold, max_pairs_per_sequence=4 if smoke else config.max_train_pairs,
        random_clip_start=True, seed=config.seed)
    scared_cal = ProposalUtilityDataset(list(SEEN), manifest["calibration_sequences"],
        coverage_threshold=config.coverage_threshold, max_pairs_per_sequence=2 if smoke else config.max_validation_pairs,
        random_clip_start=False, seed=config.seed)
    train_specimens = specimen_list(config.d4d_train_specimens)
    calibration_specimens = specimen_list(config.d4d_calibration_specimens)
    if set(train_specimens) & set(calibration_specimens):
        raise ValueError("D4D train and calibration specimens must be disjoint")
    if "specimen_3" in train_specimens or "specimen_3" in calibration_specimens:
        raise ValueError("D4D specimen_3 remains final-only")
    if not train_specimens: raise ValueError("at least one D4D train specimen is required")
    if not calibration_specimens and not config.scared_only_calibration:
        raise ValueError("empty D4D calibration requires --scared-only-calibration")
    d4d_train = D4DAnchorDataset(train_specimens, backbone="S2M2-S", max_records=4 if smoke else None)
    d4d_cal = (None if not calibration_specimens else D4DAnchorDataset(calibration_specimens, backbone="S2M2-S", max_records=2 if smoke else None))
    return scared_train, scared_cal, d4d_train, d4d_cal


def loader(dataset, config: argparse.Namespace, *, sampler=None) -> DataLoader:
    workers = min(config.workers, len(dataset))
    return DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0, drop_last=False,
        generator=torch.Generator().manual_seed(config.seed))


@torch.no_grad()
def validation(model, a2, flow, dataset, device, config) -> dict:
    model.eval(); arrays = defaultdict(list); sums = defaultdict(float); count = 0
    for cpu in loader(dataset, config):
        batch = to_device(cpu, device); evidence, _ = build_evidence(flow, batch)
        inputs, proposal = frozen_proposal_evidence(a2, batch, evidence); output = model(inputs)
        target = proposal_utility_targets(batch, proposal.disparity,
            aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
            epsilon_px=config.epsilon, coverage_threshold=config.coverage_threshold)
        losses = proposal_utility_losses(output, target, loss_config("P4"))
        for key, value in losses.items(): sums[key] += float(value)
        index = target.regression_valid.flatten().nonzero().flatten()[::64]
        cls = output.class_probability[:, 2:3] if output.class_probability is not None else output.utility
        for key, value in {"utility": target.utility, "prediction": output.utility,
                           "sigma": output.sigma, "score": cls, "helpful": target.helpful}.items():
            arrays[key].append(value.flatten()[index].float().cpu().numpy())
        count += 1
    out = {f"loss_{key}": value / max(count, 1) for key, value in sums.items()}
    values = {key: np.concatenate(value) if value else np.empty(0) for key, value in arrays.items()}
    out.update({"utility_mae": float(np.abs(values["prediction"] - values["utility"]).mean()),
                "utility_pearson": correlation(values["prediction"], values["utility"]),
                "utility_spearman": correlation(values["prediction"], values["utility"], True),
                **binary_metrics(values["score"], values["helpful"].astype(bool))})
    return out


def train(config: argparse.Namespace, *, smoke: bool) -> None:
    seed_all(config.seed); device = torch.device(config.device)
    scared_train, scared_cal, d4d_train, d4d_cal = sources(config, smoke=smoke)
    combined = MultiDomainRawErrorDataset({"SCARED-C": scared_train, "D4D": d4d_train})
    fraction = .50 if smoke else config.d4d_fraction
    sampler = DomainBalancedSampler(combined, {"SCARED-C": 1 - fraction, "D4D": fraction},
        samples_per_epoch=32 if smoke else config.samples_per_epoch, seed=config.seed)
    model = ProposalApplicabilityDetector("P4", channels=24).to(device)
    a2 = load_a2(device, config.a2_checkpoint); flow = BiDAFlowInferenceAdapter("sea_raft", device=device)
    assert sum(parameter.numel() for parameter in model.parameters()) == 15533
    assert all(not parameter.requires_grad for parameter in a2.parameters())
    assert all(not parameter.requires_grad for parameter in flow.model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    output = config.output; output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []; best = -math.inf
    history_path = output / "training_history.csv"; final_path = output / "checkpoints/final.pt"
    start = 0
    if config.resume and final_path.exists() and not smoke:
        state = torch.load(final_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start, best = int(state["epoch"]), float(state["best_validation_ap"])
        if history_path.exists(): history = list(csv.DictReader(history_path.open()))
    first = None; epochs = 2 if smoke else config.epochs
    for epoch in range(start, epochs):
        sampler.set_epoch(epoch); model.train(); sums = defaultdict(float); batches = 0
        for cpu in loader(combined, config, sampler=sampler):
            batch = to_device(cpu, device); evidence, _ = build_evidence(flow, batch)
            inputs, proposal = frozen_proposal_evidence(a2, batch, evidence); prediction = model(inputs)
            target = proposal_utility_targets(batch, proposal.disparity,
                aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
                epsilon_px=config.epsilon, coverage_threshold=config.coverage_threshold)
            target = stratified_training_targets(target, proposal.update, batch["gt"],
                maximum_pixels=config.training_pixels_per_batch)
            losses = proposal_utility_losses(prediction, target, loss_config("P4"))
            if first is None: first = float(losses["total"])
            optimizer.zero_grad(set_to_none=True); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            for key, value in losses.items(): sums[key] += float(value.detach())
            batches += 1
        scared = validation(model, a2, flow, scared_cal, device, config)
        d4d = {} if d4d_cal is None else validation(model, a2, flow, d4d_cal, device, config)
        score = float(scared.get("average_precision") or 0) if config.scared_only_calibration else .5 * (float(scared.get("average_precision") or 0) + float(d4d.get("average_precision") or 0))
        row = {"epoch": epoch + 1, "selection_ap_domain_equal": score,
               **{f"train_{k}": v / max(batches, 1) for k, v in sums.items()},
               **{f"scared_{k}": v for k, v in scared.items()}, **{f"d4d_{k}": v for k, v in d4d.items()}}
        history.append(row); write_csv(history_path, history)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                   "best_validation_ap": max(best, score), "variant": "P4", "channels": 24,
                   "a2_checkpoint": str(config.a2_checkpoint), "split": split(), "config": vars(config),
                   "frozen": {"a2": digest(config.a2_checkpoint), "sea_raft": digest(SEA_RAFT_CHECKPOINT),
                              "bida": digest(ROOT / "model_design/external_components/bidavideo.py")}}
        final_path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, final_path)
        if score > best:
            best = score; torch.save(payload, output / "checkpoints/best_validation.pt")
        print(json.dumps(row, default=str), flush=True)
    save_json(output / "split_manifest.json", split())
    save_json(output / "config.json", vars(config))
    if smoke:
        final = float(history[-1]["train_total"])
        outcome = {"initial_loss": first, "final_loss": final, "finite": math.isfinite(final),
                   "gradient_path": "P4 only; A2 and SEA-RAFT frozen", "passed": final < first}
        save_json(output / "smoke_summary.json", outcome)
        if not outcome["passed"]: raise RuntimeError(f"smoke failed: {outcome}")


@torch.no_grad()
def collect(model, a2, flow, dataset, device, config) -> dict[str, np.ndarray]:
    values = defaultdict(list)
    for cpu in loader(dataset, config):
        batch = to_device(cpu, device); evidence, _ = build_evidence(flow, batch)
        inputs, proposal = frozen_proposal_evidence(a2, batch, evidence); output = model(inputs)
        target = proposal_utility_targets(batch, proposal.disparity,
            aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
            epsilon_px=config.epsilon, coverage_threshold=config.coverage_threshold)
        mask = target.regression_valid
        for key, value in {"utility": target.utility, "raw_error": target.raw_error,
                           "proposal_error": target.proposal_error, "prediction": output.utility,
                           "sigma": output.sigma, "class": output.class_logits.argmax(1, keepdim=True),
                           "update": proposal.update}.items():
            values[key].append(value[mask].float().cpu().numpy())
    return {key: np.concatenate(value) for key, value in values.items()}


def policy_metrics(values: dict[str, np.ndarray], apply: np.ndarray, epsilon: float) -> dict:
    update = np.abs(values["update"]) > .05; changed = apply & update
    helpful, harmful = values["utility"] > epsilon, values["utility"] < -epsilon
    clean = values["raw_error"] <= .5
    output_error = np.where(apply, values["proposal_error"], values["raw_error"])
    return {"raw_epe": float(values["raw_error"].mean()), "output_epe": float(output_error.mean()),
            "gain": float(values["raw_error"].mean() - output_error.mean()), "coverage": float(changed.mean()),
            "precision": float(helpful[changed].mean()) if changed.any() else 0.0,
            "harmful_acceptance": float(harmful[changed].mean()) if changed.any() else 0.0,
            "false_update_rate": float((changed & clean).sum() / max(clean.sum(), 1)),
            "clean_degradation": float((changed & clean & harmful).sum() / max(clean.sum(), 1))}


def calibrate(config: argparse.Namespace) -> None:
    device = torch.device(config.device); checkpoint = config.output / "checkpoints/best_validation.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if digest(config.a2_checkpoint) != state["frozen"]["a2"]: raise RuntimeError("A2 hash mismatch")
    model = ProposalApplicabilityDetector("P4", channels=24).to(device).eval(); model.load_state_dict(state["model"])
    a2 = load_a2(device, config.a2_checkpoint); flow = BiDAFlowInferenceAdapter("sea_raft", device=device)
    _, scared, _, d4d = sources(config, smoke=False)
    domain = {"SCARED-C": collect(model, a2, flow, scared, device, config)}
    if d4d is not None: domain["D4D"] = collect(model, a2, flow, d4d, device, config)
    rows = []
    # The P4 class head is an auxiliary safety signal, not the utility target
    # itself.  Predeclare both its hard-veto and continuous-utility policies;
    # selection below uses only the two validation domains.
    for require_helpful_class in (True, False):
        for margin in (0., .02, .05, .10, .25):
            for sigma in (.10, .25, .50, 1., 2.):
                row = {"require_helpful_class": require_helpful_class,
                       "utility_margin_px": margin, "uncertainty_threshold_px": sigma}
                for name, value in domain.items():
                    apply = (value["prediction"] > margin) & (value["sigma"] < sigma)
                    if require_helpful_class:
                        apply &= value["class"] == 2
                    row.update({f"{name}_{key}": metric for key, metric in policy_metrics(value, apply, config.epsilon).items()})
                rows.append(row)
    def valid_row(row):
        scared_ok = row["SCARED-C_false_update_rate"] < .05 and row["SCARED-C_clean_degradation"] < .03 and row["SCARED-C_coverage"] > .002
        d4d_ok = ("D4D_coverage" not in row or (row["D4D_false_update_rate"] < .15 and row["D4D_clean_degradation"] < .10 and row["D4D_coverage"] > .002))
        return scared_ok and d4d_ok
    eligible = [row for row in rows if valid_row(row)]
    selected = max(eligible or rows, key=lambda row: (row["SCARED-C_gain"] + row.get("D4D_gain", 0.),
                                                        row["SCARED-C_precision"] + row.get("D4D_precision", 0.)))
    frozen = {"eligible": bool(eligible), "selected": selected,
              "reason": ("selected only on SCARED-C validation in an exploratory cross-specimen control; final-only data not loaded"
                         if config.scared_only_calibration else "selected only on SCARED-C validation plus D4D specimen_2; final-only data not loaded"),
              "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
              "a2_checkpoint": str(config.a2_checkpoint), "a2_sha256": digest(config.a2_checkpoint),
              "sea_raft": str(SEA_RAFT_CHECKPOINT), "sea_raft_sha256": digest(SEA_RAFT_CHECKPOINT),
              "bida_source": str(ROOT / "model_design/external_components/bidavideo.py"),
              "bida_sha256": digest(ROOT / "model_design/external_components/bidavideo.py"),
              "final_only_not_loaded": split()["forbidden_before_freeze"]}
    save_json(config.output / "calibration_summary.json", frozen)
    save_json(config.output / "frozen_manifest.json", frozen)
    write_csv(config.output / "calibration_sweep.csv", rows)
    print(json.dumps({"eligible": bool(eligible), "selected": selected}, indent=2))


def _aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows: groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, values in groups.items():
        valid = sum(int(row["valid_count"]) for row in values)
        clean = sum(int(row["clean_count"]) for row in values)
        changed = sum(int(row["changed_count"]) for row in values)
        raw_good = sum(int(row["raw_good3_count"]) for row in values)
        def weighted(name: str) -> float:
            usable = [row for row in values if math.isfinite(float(row[name])) and int(row["valid_count"]) > 0]
            count = sum(int(row["valid_count"]) for row in usable)
            return sum(float(row[name]) * int(row["valid_count"]) for row in usable) / max(count, 1)
        delta = np.asarray([float(row["refined_minus_raw_epe"]) for row in values], np.float64)
        output = {key: value for key, value in zip(keys, group)}
        output.update({"frames": len(values), "valid_count": valid,
            **{name: weighted(name) for name in ("raw_epe", "epe", "bad1", "bad3", "boundary_epe", "refined_minus_raw_epe")},
            "intervention_coverage": changed / max(valid, 1),
            "intervention_precision": sum(int(row["helpful_count"]) for row in values) / max(changed, 1),
            "false_update_rate": sum(int(row["false_update_count"]) for row in values) / max(clean, 1),
            "clean_degradation": sum(int(row["clean_degradation_count"]) for row in values) / max(clean, 1),
            "new_bad3": sum(float(row["new_bad3"]) * int(row["raw_good3_count"]) for row in values) / max(raw_good, 1),
            "frames_worsened": float((delta > 0).mean()), "worst_frame_degradation": float(delta.max()),
            "p95_frame_degradation": float(np.quantile(delta, .95))})
        result.append(output)
    return result


def _verify_frozen(config: argparse.Namespace) -> dict:
    manifest = json.loads((config.output / "frozen_manifest.json").read_text())
    checks = ((Path(manifest["checkpoint"]), manifest["checkpoint_sha256"]),
              (Path(manifest["a2_checkpoint"]), manifest["a2_sha256"]),
              (Path(manifest["sea_raft"]), manifest["sea_raft_sha256"]),
              (Path(manifest["bida_source"]), manifest["bida_sha256"]))
    for path, expected in checks:
        if digest(path) != expected: raise RuntimeError(f"frozen artifact changed: {path}")
    if not manifest["eligible"]: raise RuntimeError("selection policy was ineligible")
    return manifest


@torch.no_grad()
def final_evaluation(config: argparse.Namespace) -> None:
    """One-shot held-out seen/D4D evaluation after policy and hashes freeze."""
    completion = config.output / "final_evaluation_complete.json"
    if completion.exists(): raise RuntimeError(f"refusing repeated final evaluation: {completion}")
    manifest = _verify_frozen(config); device = torch.device(config.device)
    state = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    model = ProposalApplicabilityDetector("P4", channels=24).to(device).eval()
    model.load_state_dict(state["model"])
    a2 = load_a2(device, Path(manifest["a2_checkpoint"])); flow = BiDAFlowInferenceAdapter("sea_raft", device=device)
    policy = manifest["selected"]
    datasets = {
        "SCARED-C-final": ProposalUtilityDataset(list(SEEN), split()["final_seen_sequences"],
            coverage_threshold=config.coverage_threshold, max_pairs_per_sequence=config.max_validation_pairs,
            random_clip_start=False, seed=config.seed),
        "D4D-final-specimen3": D4DAnchorDataset(["specimen_3"], backbone="S2M2-S"),
    }
    rows: list[dict] = []; latencies: list[float] = []
    for domain, dataset in datasets.items():
        for cpu in loader(dataset, config):
            batch = to_device(cpu, device)
            if device.type == "cuda": torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if start is not None: start.record()
            evidence, _ = build_evidence(flow, batch); inputs, proposal = frozen_proposal_evidence(a2, batch, evidence)
            output = model(inputs)
            authorization = proposal_authorization_mask(output, inputs,
                utility_margin_px=float(policy["utility_margin_px"]),
                uncertainty_threshold_px=float(policy["uncertainty_threshold_px"]),
                require_helpful_class=bool(policy["require_helpful_class"]))
            if start is not None:
                end.record(); end.synchronize(); latencies.append(float(start.elapsed_time(end)) / batch["raw"].shape[0])
            raw, a2_prediction = batch["raw"], proposal.disparity
            prediction = apply_frozen_proposal(raw, a2_prediction, authorization)
            common = ((batch["gt_coverage"] > config.coverage_threshold) & batch["raw_valid"].bool()
                      & evidence["aligned_validity"].bool() & evidence["warp_support"].bool())
            boundary = boundary_mask_tensor(batch["gt"])
            methods = {"raw": (raw, torch.zeros_like(authorization)),
                       "a2_unconditional": (a2_prediction, torch.ones_like(authorization)),
                       "p4_multidomain": (prediction, authorization)}
            for name, (estimate, accepted) in methods.items():
                update = torch.where(accepted, a2_prediction - raw, torch.zeros_like(raw))
                for index in range(raw.shape[0]):
                    metric = map_metrics(estimate[index:index + 1], raw[index:index + 1], batch["gt"][index:index + 1],
                        common[index:index + 1], boundary[index:index + 1], update[index:index + 1])
                    rows.append({"dataset": domain, "backbone": batch["backbone"][index], "sequence": batch["sequence"][index],
                                 "frame_id": batch["current_frame_id"][index], "method": name,
                                 "coverage_threshold": config.coverage_threshold, **metric})
    sequence = _aggregate(rows, ("dataset", "backbone", "sequence", "method", "coverage_threshold"))
    backbone = _aggregate(rows, ("dataset", "backbone", "method", "coverage_threshold"))
    aggregate = _aggregate(rows, ("dataset", "method", "coverage_threshold"))
    lookup = {(row["dataset"], row["backbone"], row["method"]): row for row in backbone}
    seen = [lookup[("SCARED-C-final", name, "p4_multidomain")] for name in SEEN]
    d4d = lookup[("D4D-final-specimen3", "S2M2-S", "p4_multidomain")]
    def gain(row): return row["raw_epe"] - row["epe"]
    gates = {"all_seen_backbones_nonnegative_gain": all(gain(row) >= 0 for row in seen),
             "seen_safe": all(row["false_update_rate"] < .05 and row["clean_degradation"] < .03 for row in seen),
             "d4d_nonnegative_gain": gain(d4d) >= 0, "d4d_safe": d4d["false_update_rate"] < .15 and d4d["clean_degradation"] < .10,
             "d4d_nonzero_coverage": d4d["intervention_coverage"] > 0}
    verdict = "GO-to-unseen" if all(gates.values()) else "NO-GO"
    out = config.output / "final"; write_csv(out / "frame_metrics.csv", rows); write_csv(out / "sequence_metrics.csv", sequence)
    write_csv(out / "backbone_metrics.csv", backbone); write_csv(out / "aggregate_metrics.csv", aggregate)
    runtime = {"p4_plus_a2_evidence_ms_per_frame": float(np.mean(latencies)) if latencies else 0.,
               "p4_parameters": 15533, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
    save_json(out / "aggregate_summary.json", {"frozen": manifest, "metrics": aggregate, "gates": gates,
              "verdict": verdict, "runtime": runtime,
              "units": "cache-grid disparity pixels at width 180", "paired_mask": "GT coverage >0.5 & raw-valid & aligned-valid & warp-support"})
    save_json(completion, {"verdict": verdict, "gates": gates, "frozen_manifest_sha256": digest(config.output / "frozen_manifest.json"),
                           "datasets_opened_after_freeze": list(datasets), "unseen_not_loaded": ["Fast-FoundationStereo", "CREStereo", "SERV-CT", "StereoMIS"]})
    print(json.dumps({"verdict": verdict, "gates": gates}, indent=2))


def finalize_saved_final(config: argparse.Namespace) -> None:
    """Finalize a completed one-shot evaluation from compact frame CSV only."""
    completion = config.output / "final_evaluation_complete.json"
    if completion.exists(): raise RuntimeError(f"final evaluation already finalized: {completion}")
    manifest = _verify_frozen(config); out = config.output / "final"
    with (out / "frame_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows: raise RuntimeError("no saved final frame rows to finalize")
    sequence = _aggregate(rows, ("dataset", "backbone", "sequence", "method", "coverage_threshold"))
    backbone = _aggregate(rows, ("dataset", "backbone", "method", "coverage_threshold"))
    aggregate = _aggregate(rows, ("dataset", "method", "coverage_threshold"))
    lookup = {(row["dataset"], row["backbone"], row["method"]): row for row in backbone}
    seen = [lookup[("SCARED-C-final", name, "p4_multidomain")] for name in SEEN]
    d4d = lookup[("D4D-final-specimen3", "S2M2-S", "p4_multidomain")]
    gain = lambda row: row["raw_epe"] - row["epe"]
    gates = {"all_seen_backbones_nonnegative_gain": all(gain(row) >= 0 for row in seen),
             "seen_safe": all(row["false_update_rate"] < .05 and row["clean_degradation"] < .03 for row in seen),
             "d4d_nonnegative_gain": gain(d4d) >= 0, "d4d_safe": d4d["false_update_rate"] < .15 and d4d["clean_degradation"] < .10,
             "d4d_nonzero_coverage": d4d["intervention_coverage"] > 0}
    verdict = "GO-to-unseen" if all(gates.values()) else "NO-GO"
    write_csv(out / "sequence_metrics.csv", sequence); write_csv(out / "backbone_metrics.csv", backbone)
    write_csv(out / "aggregate_metrics.csv", aggregate)
    runtime = {"p4_plus_a2_evidence_ms_per_frame": None, "p4_parameters": 15533,
               "note": "one-shot inference completed before JSON serialization; reports finalized from saved frame rows without rerunning test inference"}
    save_json(out / "aggregate_summary.json", {"frozen": manifest, "metrics": aggregate, "gates": gates,
              "verdict": verdict, "runtime": runtime,
              "units": "cache-grid disparity pixels at width 180", "paired_mask": "GT coverage >0.5 & raw-valid & aligned-valid & warp-support"})
    save_json(completion, {"verdict": verdict, "gates": gates, "frozen_manifest_sha256": digest(config.output / "frozen_manifest.json"),
                           "datasets_opened_after_freeze": ["SCARED-C-final", "D4D-final-specimen3"],
                           "unseen_not_loaded": ["Fast-FoundationStereo", "CREStereo", "SERV-CT", "StereoMIS"],
                           "finalized_from_existing_frame_csv": True})
    print(json.dumps({"verdict": verdict, "gates": gates}, indent=2))


def main() -> None:
    config = args()
    if config.stage == "smoke": train(config, smoke=True)
    elif config.stage == "train": train(config, smoke=False)
    elif config.stage == "calibrate": calibrate(config)
    elif config.stage == "final": final_evaluation(config)
    else: finalize_saved_final(config)


if __name__ == "__main__": main()
