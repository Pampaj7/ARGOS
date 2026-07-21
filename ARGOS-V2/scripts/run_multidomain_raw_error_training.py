#!/usr/bin/env python3
"""Train only the ARGOS v2 S1 Raw Error Detector with supervised domain diversity."""
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
    SERVCTPairDataset, manifest_digest, stratified_raw_error_targets,
)
from model_design.data.raw_error_dataset import (  # noqa: E402
    CALIBRATION_SEQUENCES, TEST_SEQUENCES, RawErrorDataset, raw_error_targets,
)
from model_design.data.temporal_pair_dataset import (  # noqa: E402
    PRIMARY_UNSEEN_BACKBONE, SEEN_BACKBONES, TemporalPairDataset,
)
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter, SEA_RAFT_CHECKPOINT,
)
from model_design.losses.raw_error_losses import RawErrorLossConfig, raw_error_losses  # noqa: E402
from model_design.models.abstention import (  # noqa: E402
    OperatingMode, authorization_mask, calibrated_probability, fit_temperature,
)
from model_design.models.raw_error_detector import RawErrorDetector  # noqa: E402
from run_learned_t1_refiner import build_evidence  # noqa: E402
from run_raw_error_abstention import (  # noqa: E402
    A2_CHECKPOINT, aggregate_rows as legacy_aggregate_rows, atomic_checkpoint,
    binary_metrics, boundary_mask_tensor, correlation, detector_evidence, load_a2,
    map_metrics, sample_batch, to_device,
)


OUT_DEFAULT = ROOT / "results/multidomain_raw_error"
BASELINE_ROOT = ROOT / "results/raw_error_abstention/full"
BASELINE_CKPT = BASELINE_ROOT / "checkpoints/best_validation.pt"
BASELINE_MODES = BASELINE_ROOT / "operating_modes.json"
SPLIT_PATH = BASELINE_ROOT / "split_manifest.json"
BIDA_SOURCE = ROOT / "model_design/external_components/bidavideo.py"
COVERAGE_THRESHOLDS = (0.05, 0.25, 0.50, 0.90)
FOLDS = ("m1", "m2")
RATIOS = {"D1": 0.25, "D2": 0.50}


def parse_backbone_list(value: str) -> tuple[str, ...]:
    """Parse an explicit, reproducible added-domain backbone list."""
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("at least one added-domain backbone is required")
    invalid = sorted(set(items) - set(SEEN_BACKBONES))
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported added-domain backbone(s): {invalid}")
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("added-domain backbones must be unique")
    return items


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    keys = list(rows[0])
    for row in rows[1:]: keys += [key for key in row if key not in keys]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_split() -> dict:
    return json.loads(SPLIT_PATH.read_text())


def split_manifest(fold: str, datasets: dict[str, object] | None = None,
                   *, added_backbones: tuple[str, ...] = tuple(SEEN_BACKBONES)) -> dict:
    scared = base_split()
    added = ({"domain": "D4D", "train": ["specimen_1"], "calibration": ["specimen_2"],
              "final_test": ["specimen_3"], "supervision": "Zivid anchor only"}
             if fold == "m1" else
             {"domain": "SERV-CT", "train": ["honest_train__Experiment_1"],
              "calibration": ["honest_test__Experiment_2"], "final_test": [],
              "supervision": "CT-derived static/weak-replay disparity"})
    manifest = {
        "fold": fold.upper(), "seed": 20260715, "causal_pair": "t-1 -> t",
        "scared_c": {
            "train": scared["train_sequences"], "calibration": list(CALIBRATION_SEQUENCES),
            "final_test": list(TEST_SEQUENCES), "backbones": list(SEEN_BACKBONES),
            "supervision": "corrected temporal pseudo-GT",
        },
        "added_domain": {**added, "backbones": list(added_backbones)},
        "fully_unseen_domain": "SERV-CT" if fold == "m1" else "D4D",
        "unseen_backbones": [PRIMARY_UNSEEN_BACKBONE, "CREStereo"],
        "forbidden_before_freeze": (["SERV-CT", PRIMARY_UNSEEN_BACKBONE, "CREStereo", "D4D/specimen_3"]
                                    if fold == "m1" else
                                    ["D4D", PRIMARY_UNSEEN_BACKBONE, "CREStereo"]),
        "ratio_ladder": RATIOS,
        "primary_coverage": 0.50,
    }
    if datasets:
        manifest["sample_counts"] = {key: len(value) for key, value in datasets.items()}
        manifest["record_hashes"] = {
            key: manifest_digest(value.records) for key, value in datasets.items()
            if hasattr(value, "records")
        }
    return manifest


def make_sources(fold: str, args, *, smoke: bool = False):
    split = base_split()
    train_sequences = [split["train_sequences"][0]] if smoke else split["train_sequences"]
    train_backbones = [SEEN_BACKBONES[0]] if smoke else list(SEEN_BACKBONES)
    # D4D S2M2-S is geometrically audited.  The RAFT/StereoAnywhere D4D
    # caches have a measured scale mismatch and are excluded unless an
    # explicit audited protocol declares otherwise.
    added_backbones = list(args.added_backbones)
    train_pairs = 4 if smoke else args.max_train_pairs
    val_pairs = 2 if smoke else args.max_validation_pairs
    scared_train = RawErrorDataset(train_backbones, train_sequences, coverage_threshold=.5,
        max_pairs_per_sequence=train_pairs, random_clip_start=True, seed=args.seed)
    scared_cal = RawErrorDataset(list(SEEN_BACKBONES), list(CALIBRATION_SEQUENCES), coverage_threshold=.5,
        max_pairs_per_sequence=val_pairs, random_clip_start=False, seed=args.seed)
    if fold == "m1":
        added_train = D4DAnchorDataset(["specimen_1"], backbone=added_backbones,
                                       max_records=4 if smoke else None)
        added_cal = D4DAnchorDataset(["specimen_2"], backbone=added_backbones,
                                     max_records=2 if smoke else None)
    else:
        added_train = SERVCTPairDataset(["honest_train__Experiment_1"], backbone=added_backbones)
        added_cal = SERVCTPairDataset(["honest_test__Experiment_2"], backbone=added_backbones)
    return scared_train, scared_cal, added_train, added_cal


def data_loader(dataset, args, *, sampler=None, shuffle=False, batch_size=None) -> DataLoader:
    workers = min(args.workers, len(dataset))
    return DataLoader(dataset, batch_size=batch_size or args.batch_size, sampler=sampler,
        shuffle=shuffle if sampler is None else False, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0, drop_last=False,
        generator=torch.Generator().manual_seed(args.seed))


def loss_config() -> RawErrorLossConfig:
    return RawErrorLossConfig(mode="a4", false_positive_cost=5.0)


@torch.no_grad()
def validation_metrics(model, a2, flow, dataset, device, args) -> dict:
    model.eval(); totals = defaultdict(float); batches = 0; arrays = defaultdict(list)
    for cpu in data_loader(dataset, args, shuffle=False):
        batch = to_device(cpu, device); evidence, _ = build_evidence(flow, batch)
        detector_input, _ = detector_evidence(a2, batch, evidence); output = model(detector_input)
        targets = raw_error_targets(batch, epsilon_px=.5, indifference_band_px=.1, coverage_threshold=.5)
        losses = raw_error_losses(output, targets, loss_config())
        for key, value in losses.items(): totals[key] += float(value.detach())
        class_index = targets.classification_valid.flatten().nonzero().flatten()[::64]
        reg_index = targets.regression_valid.flatten().nonzero().flatten()[::64]
        arrays["p"].append(output.probability.flatten()[class_index].cpu().numpy())
        arrays["label"].append(targets.label.flatten()[class_index].cpu().numpy())
        arrays["mu"].append(output.mu.flatten()[reg_index].cpu().numpy())
        arrays["error"].append(targets.error.flatten()[reg_index].cpu().numpy())
        arrays["sigma"].append(output.sigma.flatten()[reg_index].cpu().numpy())
        batches += 1
    result = {f"loss_{key}": value / max(batches, 1) for key, value in totals.items()}
    packed = {key: np.concatenate(value) if value else np.array([]) for key, value in arrays.items()}
    result.update(binary_metrics(packed["p"], packed["label"]))
    result.update({
        "regression_mae": float(np.abs(packed["mu"] - packed["error"]).mean()),
        "pearson": correlation(packed["mu"], packed["error"]),
        "spearman": correlation(packed["mu"], packed["error"], True),
        "uncertainty_error_correlation": correlation(packed["sigma"], np.abs(packed["mu"] - packed["error"])),
    })
    return result


def train_detector(args, *, smoke: bool = False) -> None:
    seed_all(args.seed); device = torch.device(args.device)
    scared_train, scared_cal, added_train, added_cal = make_sources(args.fold, args, smoke=smoke)
    added_domain = "D4D" if args.fold == "m1" else "SERV-CT"
    combined = MultiDomainRawErrorDataset({"SCARED-C": scared_train, added_domain: added_train})
    fraction = .5 if smoke else args.added_fraction
    sampler = DomainBalancedSampler(combined, {"SCARED-C": 1-fraction, added_domain: fraction},
        samples_per_epoch=(32 if smoke else args.samples_per_epoch), seed=args.seed)
    model = RawErrorDetector("s1", channels=24).to(device)
    a2 = load_a2(device, args.a2_checkpoint); flow = BiDAFlowInferenceAdapter("sea_raft", device=device)
    assert sum(p.numel() for p in model.parameters()) == 1107
    assert all(not p.requires_grad for p in a2.parameters())
    assert all(not p.requires_grad for p in flow.model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    history_path = output / "training_history.csv"; history: list[dict] = []
    final_path = output / "checkpoints/final.pt"; best_path = output / "checkpoints/best_validation.pt"
    best, start_epoch = float("inf"), 0
    if args.resume and final_path.exists() and not smoke:
        state = torch.load(final_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_epoch, best = int(state["epoch"]), float(state["best_validation_loss"])
        if history_path.exists(): history = list(csv.DictReader(history_path.open()))
    initial_loss = None; gradient_norm = 0.0; epochs = args.smoke_epochs if smoke else args.epochs
    start_time = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch); model.train(); sums = defaultdict(float); batches = 0
        for cpu in data_loader(combined, args, sampler=sampler, batch_size=(4 if smoke else args.batch_size)):
            batch = to_device(cpu, device); evidence, _ = build_evidence(flow, batch)
            detector_input, _ = detector_evidence(a2, batch, evidence); detector = model(detector_input)
            targets = raw_error_targets(batch, epsilon_px=.5, indifference_band_px=.1, coverage_threshold=.5)
            targets = stratified_raw_error_targets(targets, pixels_per_bin=args.pixels_per_error_bin,
                                                   seed=args.seed + epoch * 10000 + batches)
            losses = raw_error_losses(detector, targets, loss_config())
            optimizer.zero_grad(set_to_none=True); losses["total"].backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); optimizer.step()
            for key, value in losses.items(): sums[key] += float(value.detach())
            batches += 1
        scared_metrics = validation_metrics(model, a2, flow, scared_cal, device, args)
        added_metrics = validation_metrics(model, a2, flow, added_cal, device, args)
        selection_loss = .5 * (scared_metrics["loss_total"] + added_metrics["loss_total"])
        row = {"epoch": epoch+1, "selection_loss_domain_equal": selection_loss,
               "gradient_norm": gradient_norm,
               **{f"train_{key}": value/max(batches,1) for key,value in sums.items()},
               **{f"scared_cal_{key}": value for key,value in scared_metrics.items()},
               **{f"added_cal_{key}": value for key,value in added_metrics.items()}}
        history.append(row); write_csv(history_path, history)
        if initial_loss is None:
            initial_loss = float(row["train_total"])
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch+1,
            "best_validation_loss": min(best, selection_loss), "architecture": "s1", "channels": 24,
            "fold": args.fold, "added_fraction": fraction, "config": clean(vars(args)),
            "split_manifest": split_manifest(args.fold, added_backbones=args.added_backbones), "a2_checkpoint": str(args.a2_checkpoint),
            "loss_config": asdict(loss_config()), "sampler_epoch": epoch+1,
        }
        atomic_checkpoint(final_path, payload)
        if selection_loss < best:
            best = selection_loss; payload["best_validation_loss"] = best; atomic_checkpoint(best_path, payload)
        print(json.dumps(clean(row)), flush=True)
    manifest = split_manifest(args.fold, {"scared_train": scared_train, "scared_calibration": scared_cal,
                                          "added_train": added_train, "added_calibration": added_cal},
                              added_backbones=args.added_backbones)
    manifest["sampling"] = {"SCARED-C": 1-fraction, added_domain: fraction,
                            "samples_per_epoch": len(sampler), "domain_counts": sampler.domain_counts(),
                            "pixel_strata": ["clean<=0.5", "moderate(0.5,3]", "large>3"]}
    save_json(output / "split_manifest.json", manifest); save_json(output / "config.json", vars(args))
    runtime = {"training_seconds": time.perf_counter()-start_time, "detector_parameters": 1107,
               "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
    save_json(output / "runtime_summary.json", runtime)
    if smoke:
        final_loss = min(float(row["train_total"]) for row in history[1:]) if len(history) > 1 else float(history[-1]["train_total"])
        summary = {"initial_loss": initial_loss, "final_loss": final_loss,
                   "loss_reduction": (initial_loss-final_loss)/max(initial_loss,1e-8),
                   "gradient_nonzero": gradient_norm > 0, "finite": math.isfinite(final_loss),
                   "passed": final_loss < initial_loss and gradient_norm > 0 and math.isfinite(final_loss)}
        save_json(output / "smoke_summary.json", summary)
        if not summary["passed"]: raise RuntimeError(f"smoke failed: {summary}")


def collect_domain_samples(model, a2, flow, dataset, device, args) -> dict[str, np.ndarray]:
    from run_raw_error_abstention import collect_samples
    return collect_samples(model, a2, flow, dataset, device, args)


def balanced_concat(parts: dict[str, dict[str, np.ndarray]], seed: int) -> dict[str, np.ndarray]:
    count = min(len(value["error"]) for value in parts.values())
    rng = np.random.default_rng(seed); output = defaultdict(list)
    for domain in sorted(parts):
        values = parts[domain]; index = rng.choice(len(values["error"]), size=count, replace=False)
        for key, value in values.items(): output[key].append(value[index])
        output["domain"].append(np.full(count, domain))
    return {key: np.concatenate(value) for key,value in output.items()}


def mode_metrics(samples: dict[str,np.ndarray], temperature: float, p: float, mu: float, sigma: float) -> dict:
    probability = 1/(1+np.exp(-samples["logits"]/temperature))
    apply = ((probability>=p)&(samples["mu"]>=mu)&(samples["sigma"]<=sigma)
             &(np.abs(samples["proposal_update"])<=3))
    changed = apply & (np.abs(samples["proposal_update"])>.05)
    raw_error, a2_error = samples["error"], samples["a2_error"]
    refined = np.where(apply,a2_error,raw_error); clean=raw_error<=.5
    harmful=changed&(a2_error>raw_error+.02); helpful=changed&(a2_error+.02<raw_error)
    a2_gain=float(raw_error.mean()-a2_error.mean()); gain=float(raw_error.mean()-refined.mean())
    return {"raw_epe":float(raw_error.mean()),"a2_epe":float(a2_error.mean()),"refined_epe":float(refined.mean()),
        "epe_gain":gain,"retained_a2_gain":gain/max(a2_gain,1e-8),"intervention_coverage":float(changed.mean()),
        "intervention_precision":float(helpful.sum()/max(changed.sum(),1)),
        "false_update_rate":float((changed&clean).sum()/max(clean.sum(),1)),
        "clean_degradation":float((harmful&clean).sum()/max(clean.sum(),1))}


def calibrate_candidate(candidate: Path, fold: str, args) -> dict:
    state=torch.load(candidate/"checkpoints/best_validation.pt",map_location="cpu",weights_only=False)
    device=torch.device(args.device); model=RawErrorDetector("s1",channels=24).to(device)
    model.load_state_dict(state["model"]); model.eval(); a2=load_a2(device, args.a2_checkpoint)
    flow=BiDAFlowInferenceAdapter("sea_raft",device=device)
    _train,scared_cal,_added_train,added_cal=make_sources(fold,args)
    added_name="D4D" if fold=="m1" else "SERV-CT"
    samples={"SCARED-C":collect_domain_samples(model,a2,flow,scared_cal,device,args),
             added_name:collect_domain_samples(model,a2,flow,added_cal,device,args)}
    combined=balanced_concat(samples,args.seed)
    temperature=fit_temperature(torch.from_numpy(combined["logits"]),torch.from_numpy(combined["label"]),
        torch.from_numpy(combined["class_valid"]),split="validation")
    rows=[]
    for p in (.50,.65,.80,.90,.95):
      for mu in (.25,.50,1.0):
       for sigma in (.25,.50,1.0,2.0):
        row={"probability_threshold":p,"error_threshold_px":mu,"uncertainty_threshold_px":sigma}
        for domain,part in samples.items():
            for key,value in mode_metrics(part,temperature,p,mu,sigma).items(): row[f"{domain}_{key}"]=value
        rows.append(row)
    def constraints(row):
        s_ok=(row["SCARED-C_false_update_rate"]<.05 and row["SCARED-C_clean_degradation"]<.03
              and row["SCARED-C_retained_a2_gain"]>=.70)
        a_ok=(row[f"{added_name}_false_update_rate"]<.15 and row[f"{added_name}_clean_degradation"]<.10
              and row[f"{added_name}_epe_gain"]>=-1e-3)
        return s_ok and a_ok
    eligible=[row for row in rows if constraints(row) and row["SCARED-C_intervention_coverage"]>0]
    def score(row): return (row["SCARED-C_epe_gain"]+row[f"{added_name}_epe_gain"],
                            row["SCARED-C_intervention_precision"]+row[f"{added_name}_intervention_precision"])
    if eligible: balanced=max(eligible,key=score); eligible_status=True
    else:
        def violation(row):
            return (max(0,row["SCARED-C_false_update_rate"]-.05)+max(0,row["SCARED-C_clean_degradation"]-.03)
                    +max(0,.70-row["SCARED-C_retained_a2_gain"])+max(0,row[f"{added_name}_false_update_rate"]-.15)
                    +max(0,row[f"{added_name}_clean_degradation"]-.10)+max(0,-row[f"{added_name}_epe_gain"]))
        balanced=min(rows,key=lambda row:(violation(row),-score(row)[0])); eligible_status=False
    ultra=min(rows,key=lambda row:(row["SCARED-C_false_update_rate"]+row[f"{added_name}_false_update_rate"],
                                   -row["SCARED-C_intervention_precision"]-row[f"{added_name}_intervention_precision"]))
    modes={"balanced":OperatingMode("balanced",balanced["probability_threshold"],balanced["error_threshold_px"],balanced["uncertainty_threshold_px"]),
           "ultra_safe":OperatingMode("ultra_safe",ultra["probability_threshold"],ultra["error_threshold_px"],ultra["uncertainty_threshold_px"])}
    summary={"candidate":str(candidate),"fold":fold,"temperature":temperature,"eligible":eligible_status,
             "balanced":balanced,"modes":{key:value.as_dict() for key,value in modes.items()},
             "checkpoint_validation_loss":float(state["best_validation_loss"])}
    write_csv(candidate/"calibration_sweep.csv",rows); save_json(candidate/"calibration_summary.json",summary)
    return summary


def freeze_selection(args) -> None:
    fold_root=args.output/args.fold.upper(); candidates=[]
    for ratio_name in RATIOS:
        candidate=fold_root/ratio_name
        if not (candidate/"checkpoints/best_validation.pt").exists(): raise FileNotFoundError(candidate)
        candidates.append(calibrate_candidate(candidate,args.fold,args))
    eligible=[item for item in candidates if item["eligible"]]
    pool=eligible or candidates
    added="D4D" if args.fold=="m1" else "SERV-CT"
    selected=max(pool,key=lambda item:(item["balanced"]["SCARED-C_epe_gain"]+item["balanced"][f"{added}_epe_gain"],
                                       -item["checkpoint_validation_loss"]))
    frozen=fold_root/"frozen"
    if frozen.exists(): raise FileExistsError(f"refusing to overwrite frozen selection: {frozen}")
    (frozen/"checkpoints").mkdir(parents=True)
    source=Path(selected["candidate"])/"checkpoints/best_validation.pt"
    shutil.copy2(source,frozen/"checkpoints/best_validation.pt")
    modes={"temperature":selected["temperature"],"modes":selected["modes"],
           "provenance":"SCARED-C calibration + added-domain calibration only"}
    save_json(frozen/"operating_modes.json",modes); save_json(fold_root/"ratio_selection.json",{
        "candidates":candidates,"selected":selected["candidate"],"selection_used_unseen":False})
    artifacts={"detector":frozen/"checkpoints/best_validation.pt","modes":frozen/"operating_modes.json",
               "a2":args.a2_checkpoint,"bida_source":BIDA_SOURCE,"sea_raft":SEA_RAFT_CHECKPOINT,"split":SPLIT_PATH}
    manifest={"status":"frozen before unseen-domain/backbone loading","fold":args.fold,
              "forbidden_not_loaded":split_manifest(args.fold, added_backbones=args.added_backbones)["forbidden_before_freeze"],
              "artifacts":{key:{"path":str(path),"sha256":sha256(path)} for key,path in artifacts.items()}}
    save_json(frozen/"frozen_manifest.json",manifest)
    print(json.dumps(clean({"frozen":str(frozen),"selected":selected}),indent=2))


def load_detector(path: Path, device: torch.device):
    state=torch.load(path,map_location="cpu",weights_only=False)
    model=RawErrorDetector(state["architecture"],channels=int(state["channels"])).to(device)
    model.load_state_dict(state["model"]); return model.eval().requires_grad_(False)


def load_modes(path: Path):
    value=json.loads(path.read_text())
    return float(value["temperature"]),{key:OperatingMode(**mode) for key,mode in value["modes"].items()}


@torch.no_grad()
def evaluate_dataset(name,dataset,new_model,new_temp,new_mode,base_model,base_temp,base_mode,a2,flow,device,args):
    rows=[]; detector_store=defaultdict(list); latency=[]
    for cpu in data_loader(dataset,args,shuffle=False):
        batch=to_device(cpu,device)
        if device.type=="cuda": torch.cuda.synchronize(device)
        total_start=time.perf_counter()
        evidence,_=build_evidence(flow,batch); inp,proposal=detector_evidence(a2,batch,evidence)
        if device.type=="cuda": torch.cuda.synchronize(device)
        start=time.perf_counter(); new_out=new_model(inp)
        if device.type=="cuda": torch.cuda.synchronize(device)
        latency.append((time.perf_counter()-start)*1000/batch["raw"].shape[0])
        base_out=base_model(inp)
        new_auth=authorization_mask(new_out,mode=new_mode,temperature=new_temp,aligned_valid=evidence["aligned_validity"],
            warp_support=evidence["warp_support"],proposal_update=proposal.update)
        base_auth=authorization_mask(base_out,mode=base_mode,temperature=base_temp,aligned_valid=evidence["aligned_validity"],
            warp_support=evidence["warp_support"],proposal_update=proposal.update)
        if device.type=="cuda": torch.cuda.synchronize(device)
        total_ms=(time.perf_counter()-total_start)*1000/batch["raw"].shape[0]
        latency[-1]=(latency[-1],total_ms)
        oracle=(proposal.disparity-batch["gt"]).abs()<(batch["raw"]-batch["gt"]).abs()
        updates={"raw":torch.zeros_like(proposal.update),"a2_no_authorization":proposal.update,
                 "m0_scared_only":torch.where(base_auth,proposal.update,torch.zeros_like(proposal.update)),
                 "multidomain":torch.where(new_auth,proposal.update,torch.zeros_like(proposal.update)),
                 "oracle_authorization":torch.where(oracle,proposal.update,torch.zeros_like(proposal.update))}
        boundary=boundary_mask_tensor(batch["gt"]); aligned=evidence["aligned_past_disparity"]
        for threshold in COVERAGE_THRESHOLDS:
            common=(batch["gt_coverage"]>threshold)&batch["raw_valid"].bool()&evidence["aligned_validity"]&evidence["warp_support"]
            for method,update in updates.items():
                prediction=batch["raw"]+update
                for index in range(batch["raw"].shape[0]):
                    geometry=map_metrics(prediction[index:index+1],batch["raw"][index:index+1],batch["gt"][index:index+1],
                        common[index:index+1],boundary[index:index+1],update[index:index+1])
                    support=common[index:index+1]
                    raw_t=(batch["raw"][index:index+1]-aligned[index:index+1]).abs()
                    refined_t=(prediction[index:index+1]-aligned[index:index+1]).abs()
                    rows.append({"dataset":name,"domain":batch.get("domain",["SCARED-C"]*len(batch["backbone"]))[index],
                        "backbone":batch["backbone"][index],"sequence":batch["sequence"][index],
                        "frame_id":batch["current_frame_id"][index],"coverage_threshold":threshold,"method":method,
                        "raw_mc_temporal_error":float(raw_t[support].mean()) if support.any() else math.nan,
                        "refined_mc_temporal_error":float(refined_t[support].mean()) if support.any() else math.nan,
                        **geometry})
        targets=raw_error_targets(batch,epsilon_px=.5,indifference_band_px=.1,coverage_threshold=.5)
        valid=targets.regression_valid&evidence["aligned_validity"]&evidence["warp_support"]
        index=valid.flatten().nonzero().flatten()[::64]
        class_valid=targets.classification_valid.flatten()[index].cpu().numpy().astype(bool)
        for key,value in {"probability":calibrated_probability(new_out.logits,new_temp),"label":targets.label,"mu":new_out.mu,
                          "sigma":new_out.sigma,"error":targets.error,"class_valid":targets.classification_valid}.items():
            detector_store[key].append(value.flatten()[index].float().cpu().numpy())
    packed={key:np.concatenate(value) for key,value in detector_store.items()}
    cv=packed["class_valid"].astype(bool); dm={**binary_metrics(packed["probability"][cv],packed["label"][cv]),
        "raw_error_mae":float(np.abs(packed["mu"]-packed["error"]).mean()),
        "mu_error_pearson":correlation(packed["mu"],packed["error"]),
        "mu_error_spearman":correlation(packed["mu"],packed["error"],True),
        "uncertainty_error_correlation":correlation(packed["sigma"],np.abs(packed["mu"]-packed["error"]))}
    return rows,dm,latency


def aggregate_evaluation(rows:list[dict],keys=("dataset","backbone","method","coverage_threshold")):
    groups=defaultdict(list)
    for row in rows: groups[tuple(row[key] for key in keys)].append(row)
    output=[]
    for group_key,group in groups.items():
        valid=sum(row["valid_count"] for row in group); clean=sum(row["clean_count"] for row in group)
        changed=sum(row["changed_count"] for row in group); raw_good=sum(row["raw_good3_count"] for row in group)
        def weighted(metric):
            eligible=[row for row in group if row["valid_count"] > 0 and math.isfinite(row[metric])]
            denominator=sum(row["valid_count"] for row in eligible)
            return sum(row[metric]*row["valid_count"] for row in eligible)/denominator if denominator else math.nan
        degr=np.array([row["refined_minus_raw_epe"] for row in group
                       if math.isfinite(row["refined_minus_raw_epe"])])
        item={key:value for key,value in zip(keys,group_key)}
        item.update({"frames":len(group),"valid_count":valid,"raw_epe":weighted("raw_epe"),"epe":weighted("epe"),
            "refined_minus_raw_epe":weighted("refined_minus_raw_epe"),"bad1":weighted("bad1"),"bad3":weighted("bad3"),
            "boundary_epe":weighted("boundary_epe"),"new_bad3":sum(r["new_bad3"]*r["raw_good3_count"] for r in group)/max(raw_good,1),
            "intervention_coverage":changed/max(valid,1),"intervention_precision":sum(r["helpful_count"] for r in group)/max(changed,1),
            "false_update_rate":sum(r["false_update_count"] for r in group)/max(clean,1),
            "clean_degradation":sum(r["clean_degradation_count"] for r in group)/max(clean,1),
            "mean_clean_update":sum(r["mean_update_magnitude_clean"]*r["clean_count"] for r in group)/max(clean,1),
            "frames_worsened":float((degr>0).mean()) if degr.size else math.nan,
            "worst_frame_degradation":float(degr.max()) if degr.size else math.nan,
            "p95_frame_degradation":float(np.quantile(degr,.95)) if degr.size else math.nan,
            "catastrophic_tail_p99":float(np.quantile(degr,.99)) if degr.size else math.nan,
            "raw_mc_temporal_error":weighted("raw_mc_temporal_error"),
            "refined_mc_temporal_error":weighted("refined_mc_temporal_error")})
        output.append(item)
    return output


def verify_frozen_manifest(frozen:Path):
    manifest=json.loads((frozen/"frozen_manifest.json").read_text())
    for item in manifest["artifacts"].values():
        if sha256(Path(item["path"]))!=item["sha256"]: raise RuntimeError(f"frozen hash mismatch: {item['path']}")
    return manifest


def generate_final_report(fold_root: Path, fold: str, rows: list[dict], aggregate: list[dict],
                          detector_rows: list[dict], runtime: dict) -> None:
    out = fold_root / "evaluation"
    per_dataset = aggregate_evaluation(rows, ("dataset", "method", "coverage_threshold"))
    write_csv(out / "per_dataset.csv", per_dataset)
    specimen_rows=[]
    for row in rows:
        value=dict(row)
        value["specimen"]=(row["sequence"].split("__")[0]
                           if row["dataset"].startswith("D4D") else row["sequence"])
        specimen_rows.append(value)
    write_csv(out / "per_specimen.csv", aggregate_evaluation(
        specimen_rows, ("dataset", "specimen", "method", "coverage_threshold")))
    primary = {(row["dataset"], row["method"]): row for row in per_dataset
               if float(row["coverage_threshold"]) == .5}
    selected = [row for row in per_dataset if float(row["coverage_threshold"]) == .5
                and row["method"] == "multidomain"]
    save_json(out / "safety_summary.json", {row["dataset"]: row for row in selected})

    verdicts = {}
    for row in selected:
        dataset = row["dataset"]
        raw = primary[(dataset, "raw")]
        baseline = primary[(dataset, "m0_scared_only")]
        gain = raw["epe"] - row["epe"]
        baseline_gain = raw["epe"] - baseline["epe"]
        retention = gain / max(baseline_gain, 1e-8) if baseline_gain > 0 else None
        if dataset == "SCARED-C-test":
            passed = retention is not None and retention >= .70 and row["false_update_rate"] < .05 and row["clean_degradation"] < .03
            reason = f"M0 gain retention {retention:.1%}; false update {row['false_update_rate']:.1%}; clean degradation {row['clean_degradation']:.1%}."
        elif dataset in {"Fast-FoundationStereo", "CREStereo"}:
            passed = gain >= -1e-3 and row["false_update_rate"] < .05 and row["clean_degradation"] < .03
            reason = f"gain {gain:+.4f}px; false update {row['false_update_rate']:.1%}; clean degradation {row['clean_degradation']:.1%}."
        elif dataset.startswith("D4D"):
            passed = gain >= -1e-3 and row["false_update_rate"] < .15 and row["clean_degradation"] < .10
            reason = f"gain {gain:+.4f}px; false update {row['false_update_rate']:.1%}; clean degradation {row['clean_degradation']:.1%}."
        else:
            passed = gain >= -1e-3 and row["false_update_rate"] < .15 and row["clean_degradation"] < .10
            reason = f"gain {gain:+.4f}px; false update {row['false_update_rate']:.1%}; clean degradation {row['clean_degradation']:.1%}; coverage {row['intervention_coverage']:.1%}."
        verdicts[dataset] = {"verdict": "GO" if passed else "NO-GO", "reason": reason,
                             "epe_gain": gain, "m0_gain_retention": retention}
    unseen_prefix = "SERV-CT" if fold == "m1" else "D4D-unseen"
    unseen_key = next((key for key in verdicts if key.startswith(unseen_prefix)), None)
    hypothesis = bool(unseen_key and verdicts[unseen_key]["verdict"] == "GO"
                      and primary[(unseen_key, "multidomain")]["intervention_coverage"] > 0
                      and primary[(unseen_key, "multidomain")]["false_update_rate"]
                      < primary[(unseen_key, "m0_scared_only")]["false_update_rate"])
    verdicts["overall"] = {
        "verdict": "GO" if hypothesis and all(value["verdict"] == "GO" for key, value in verdicts.items() if key != "overall") else "NO-GO",
        "scientific_hypothesis_supported": hypothesis,
        "reason": "Multi-domain exposure must improve the fully unseen domain with nonzero coverage while preserving every required seen/backbone safety check.",
    }
    save_json(out / "verdicts.json", verdicts)

    lines = [f"# ARGOS v2 — Multi-domain Raw Error Detector ({fold.upper()})", "",
             f"Overall verdict: **{verdicts['overall']['verdict']}**.", "",
             "All values below are cache-grid metrics at fractional GT coverage 0.50. No OOD sample was loaded before detector, temperature and thresholds were frozen.", "",
             "| Dataset | Method | Raw EPE | Output EPE | Gain | False update | Clean degradation | Coverage | Precision |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in sorted({row["dataset"] for row in selected}):
        raw = primary[(dataset, "raw")]
        for method in ("m0_scared_only", "multidomain"):
            row = primary[(dataset, method)]
            lines.append(f"| {dataset} | {method} | {raw['epe']:.6f} | {row['epe']:.6f} | {raw['epe']-row['epe']:+.6f} | {row['false_update_rate']:.2%} | {row['clean_degradation']:.2%} | {row['intervention_coverage']:.2%} | {row['intervention_precision']:.2%} |")
    lines += ["", "## Frozen protocol", "",
              "- Trainable component: unchanged S1 Raw Error Detector (1,107 parameters).",
              "- Frozen: disparity caches, SEA-RAFT, canonical BiDA warp, A2 proposal and bounded update.",
              "- D4D supervision: curated Zivid anchor only; no temporal propagation and no prediction-derived GT.",
              ("- M1 selection/calibration: held-out SCARED-C plus D4D specimen 2 only; SERV-CT and unseen backbones are final-only."
               if fold == "m1" else
               "- M2 selection/calibration: held-out SCARED-C plus SERV-CT Experiment 2 only; all D4D specimens and unseen backbones are final-only."),
              "", "## Exact commands", "", "```bash",
              "PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python",
              f"$PY scripts/run_multidomain_raw_error_training.py --stage train --fold {fold} --added-fraction 0.25 --output results/multidomain_raw_error --device cuda:0 --workers 16 --batch-size 16 --epochs 5 --samples-per-epoch 2048 --pixels-per-error-bin 512",
              f"$PY scripts/run_multidomain_raw_error_training.py --stage train --fold {fold} --added-fraction 0.50 --output results/multidomain_raw_error --device cuda:1 --workers 16 --batch-size 16 --epochs 5 --samples-per-epoch 2048 --pixels-per-error-bin 512",
              f"$PY scripts/run_multidomain_raw_error_training.py --stage select --fold {fold} --output results/multidomain_raw_error --device cuda:0 --workers 16 --batch-size 16",
              f"$PY scripts/run_multidomain_raw_error_training.py --stage final --fold {fold} --output results/multidomain_raw_error --device cuda:0 --workers 16 --batch-size 16",
              "```", ""]
    (out / "README.md").write_text("\n".join(lines))


def final_evaluation(args) -> None:
    fold_root=args.output/args.fold.upper(); frozen=fold_root/"frozen"; completion=fold_root/"final_evaluation_complete.json"
    if completion.exists(): raise RuntimeError(f"one-shot final evaluation already exists: {completion}")
    manifest=verify_frozen_manifest(frozen); device=torch.device(args.device)
    new_model=load_detector(frozen/"checkpoints/best_validation.pt",device); new_temp,new_modes=load_modes(frozen/"operating_modes.json")
    base_model=load_detector(BASELINE_CKPT,device); base_temp,base_modes=load_modes(BASELINE_MODES)
    a2=load_a2(device,args.a2_checkpoint); flow=BiDAFlowInferenceAdapter("sea_raft",device=device)
    # All constructors below occur after serialized policy/hash verification.
    datasets={
      "SCARED-C-test":RawErrorDataset(list(SEEN_BACKBONES),list(TEST_SEQUENCES),coverage_threshold=.5,max_pairs_per_sequence=args.max_validation_pairs,random_clip_start=False,seed=args.seed),
      "Fast-FoundationStereo":TemporalPairDataset([PRIMARY_UNSEEN_BACKBONE],list(TEST_SEQUENCES),coverage_threshold=.5,max_pairs_per_sequence=args.max_validation_pairs,random_clip_start=False,seed=args.seed),
      "CREStereo":TemporalPairDataset(["CREStereo"],list(TEST_SEQUENCES),coverage_threshold=.5,max_pairs_per_sequence=args.max_validation_pairs,random_clip_start=False,seed=args.seed),
    }
    if args.fold=="m1":
        datasets["D4D-heldout-specimen3"]=D4DAnchorDataset(["specimen_3"], backbone=list(args.added_backbones))
        datasets["SERV-CT-unseen"]=SERVCTPairDataset(["honest_train__Experiment_1","honest_test__Experiment_2"], backbone=list(SEEN_BACKBONES))
    else:
        datasets["SERV-CT-seen-calibration"]=SERVCTPairDataset(["honest_test__Experiment_2"], backbone=list(SEEN_BACKBONES))
        datasets["D4D-unseen"]=D4DAnchorDataset(["specimen_1","specimen_2","specimen_3"], backbone=list(SEEN_BACKBONES))
    rows=[]; detector_rows=[]; latencies=[]
    for name,dataset in datasets.items():
        part,detector,latency=evaluate_dataset(name,dataset,new_model,new_temp,new_modes["balanced"],base_model,base_temp,
            base_modes["balanced"],a2,flow,device,args)
        rows+=part; detector_rows.append({"dataset":name,**detector}); latencies+=latency
        print(json.dumps({"completed":name,"frames":len(dataset)}),flush=True)
    sequence=aggregate_evaluation(rows,("dataset","backbone","sequence","method","coverage_threshold"))
    aggregate=aggregate_evaluation(rows)
    out=fold_root/"evaluation"; write_csv(out/"frame_metrics.csv",rows); write_csv(out/"sequence_metrics.csv",sequence)
    write_csv(out/"per_dataset_backbone.csv",aggregate); write_csv(out/"detector_metrics.csv",detector_rows)
    detector_latency=np.asarray([value[0] for value in latencies]); total_latency=np.asarray([value[1] for value in latencies])
    runtime={"detector_latency_ms_mean":float(detector_latency.mean()),"detector_latency_ms_p95":float(np.quantile(detector_latency,.95)),
             "total_argos_latency_ms_mean":float(total_latency.mean()),"total_argos_latency_ms_p95":float(np.quantile(total_latency,.95)),
             "detector_parameters":1107,"peak_gpu_bytes":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0}
    save_json(out/"runtime_summary.json",runtime)
    save_json(completion,{"fold":args.fold,"completed_datasets":list(datasets),"frozen_manifest_sha256":sha256(frozen/"frozen_manifest.json"),
                          "no_post_unseen_tuning":True})
    save_json(out/"aggregate_summary.json",{"fold":args.fold,"frozen":manifest,"metrics":aggregate,"detector":detector_rows,"runtime":runtime})
    generate_final_report(fold_root,args.fold,rows,aggregate,detector_rows,runtime)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("smoke","train","select","final"),required=True)
    parser.add_argument("--fold",choices=FOLDS,default="m1")
    parser.add_argument("--output",type=Path,default=OUT_DEFAULT)
    parser.add_argument("--added-fraction",type=float,choices=(.25,.50),default=.25)
    parser.add_argument("--epochs",type=int,default=5); parser.add_argument("--smoke-epochs",type=int,default=8)
    parser.add_argument("--batch-size",type=int,default=16); parser.add_argument("--workers",type=int,default=min(32,os.cpu_count() or 8))
    parser.add_argument("--max-train-pairs",type=int,default=256); parser.add_argument("--max-validation-pairs",type=int,default=160)
    parser.add_argument("--samples-per-epoch",type=int,default=2048); parser.add_argument("--pixels-per-error-bin",type=int,default=512)
    parser.add_argument("--sample-pixels-per-frame",type=int,default=2048)
    parser.add_argument("--added-backbones", type=parse_backbone_list,
                        default=("S2M2-S",),
                        help="comma-separated, audited backbones for the added training domain")
    parser.add_argument("--a2-checkpoint", type=Path, default=A2_CHECKPOINT,
                        help="frozen A2 proposal checkpoint; never train through this artifact")
    parser.add_argument("--learning-rate",type=float,default=2e-3); parser.add_argument("--weight-decay",type=float,default=1e-4)
    parser.add_argument("--seed",type=int,default=20260715); parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True)
    # Compatibility fields required by reused sample/target utilities.
    parser.set_defaults(epsilon=.5,indifference_band=.1,coverage_threshold=.5,loss_mode="a4",false_positive_cost=5.0)
    args=parser.parse_args()
    if args.stage=="train":
        name="D1" if args.added_fraction==.25 else "D2"; args.output=args.output/args.fold.upper()/name
    elif args.stage=="smoke": args.output=args.output/"smoke"
    return args


def main():
    args=parse_args()
    if args.stage=="smoke": train_detector(args,smoke=True)
    elif args.stage=="train": train_detector(args)
    elif args.stage=="select": freeze_selection(args)
    else: final_evaluation(args)


if __name__=="__main__": main()
