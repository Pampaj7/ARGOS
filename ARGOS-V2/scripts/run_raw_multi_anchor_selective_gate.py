#!/usr/bin/env python3
"""Train and evaluate ARGOS v2 post-hoc veto gates for frozen raw-anchor proposals."""
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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from model_design.models.raw_multi_anchor_refiner import PROVENANCE_RAW, RAW_ANCHOR_AGES  # noqa: E402
from model_design.models.raw_multi_anchor_selective_gate import (  # noqa: E402
    GATE_FEATURE_CHANNELS, GATE_FEATURE_NAMES, LinearHarmGate, SelectiveRiskGate,
    apply_veto, build_gate_features, frozen_proposal,
)
from run_raw_multi_anchor_temporal_refiner import (  # noqa: E402
    TEST, TRAIN, VALIDATION, aggregate_metrics, build_ram_bank, evidence_from_batch,
    grouped_metrics, loader, sha256, split_manifest, to_device, write_csv,
)


OUTPUT = ROOT / "results/raw_multi_anchor_selective_gate"
FROZEN_CHECKPOINT = ROOT / "results/raw_multi_anchor_temporal_refiner/soft_fusion/checkpoints/best_validation.pt"
FROZEN_POLICY = ROOT / "results/raw_multi_anchor_temporal_refiner/soft_fusion/frozen_policy.json"
EXPECTED_FROZEN_SHA256 = "40526a32ef6e9a62a3ea2b59e6751a60c441b8190f9b96522e3b12b35895d5cd"
FIXED_H4_EPE = 0.5172594986037409
RAW_BANK_ORACLE_GAIN = 0.139524
MODEL_NAMES = ("logistic", "harm_mlp", "joint_gain_risk")
MARGINS = (0.0, 0.05, 0.10)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "overfit", "smoke", "train", "calibrate", "evaluate", "summarize"), required=True)
    parser.add_argument("--model", choices=MODEL_NAMES + ("all",), default="all")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    parser.add_argument("--flow-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--crop-height", type=int, default=64)
    parser.add_argument("--crop-width", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--margin", type=float, default=.10)
    parser.add_argument("--coverage-threshold", type=float, default=.50)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sample-stride", type=int, default=8)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def clean(value):
    if isinstance(value, dict): return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def frozen_policy() -> dict:
    record = json.loads(FROZEN_POLICY.read_text())
    if record["checkpoint_sha256"] != EXPECTED_FROZEN_SHA256 or sha256(FROZEN_CHECKPOINT) != EXPECTED_FROZEN_SHA256:
        raise RuntimeError("frozen raw multi-anchor checkpoint hash mismatch")
    return record["policy"]


def load_frozen(device: torch.device):
    from model_design.models.raw_multi_anchor_refiner import RawMultiAnchorRefiner
    state = torch.load(FROZEN_CHECKPOINT, map_location="cpu", weights_only=False)
    model = RawMultiAnchorRefiner(int(state["channels"]), int(state["blocks"])).to(device)
    model.load_state_dict(state["model"]); model.eval().requires_grad_(False)
    return model, state


def gate_model(name: str, config: argparse.Namespace, device: torch.device):
    if name == "logistic": model = LinearHarmGate()
    else: model = SelectiveRiskGate(hidden=config.hidden, depth=config.depth, joint=name == "joint_gain_risk")
    return model.to(device)


def model_root(config: argparse.Namespace, name: str) -> Path:
    return config.output / name


def proposal_tensors(batch, evidence, frozen, policy):
    with torch.no_grad():
        base_output = frozen(evidence)
        proposal = frozen_proposal(evidence, base_output, probability_threshold=policy["probability_threshold"],
                                   utility_threshold_px=policy["utility_threshold_px"])
        features = build_gate_features(evidence, proposal)
    gt = batch["gt"].float(); raw_error = (evidence.raw - gt).abs(); proposed_error = (proposal.proposed - gt).abs()
    delta = raw_error - proposed_error
    supervised = (batch["gt_coverage"].float() > .50) & batch["raw_valid"].bool() & proposal.eligible
    return proposal, features, delta, raw_error, proposed_error, supervised


def training_loss(output, delta, mask, margin: float, joint: bool):
    target = (delta < -margin).float()
    logits = output.harm_logit[mask]; labels = target[mask]
    if logits.numel() == 0: return output.harm_logit.sum() * 0, {"harm": 0.0, "delta": 0.0}
    positive = labels.sum(); negative = labels.numel() - positive
    # False-safe errors are the expensive error, hence harmful labels receive extra weight.
    pos_weight = ((negative / positive.clamp_min(1)).clamp(.5, 4.0) * 2.0).detach()
    harm = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    regression = F.huber_loss(output.predicted_gain[mask], delta[mask], delta=.25) if joint else harm.new_zeros(())
    return harm + .25 * regression, {"harm": float(harm.detach()), "delta": float(regression.detach())}


@torch.inference_mode()
def sampled_diagnostics(model, dataset, frozen, state, config, margin: float) -> dict:
    device = torch.device(config.device); policy = frozen_policy(); labels=[]; probabilities=[]; gains=[]; predictions=[]
    ages, provenance = state["ages"], state["provenance"]
    model.eval()
    for cpu in loader(dataset, config, training=False):
        batch=to_device(cpu,device); evidence=evidence_from_batch(batch,ages,provenance)
        proposal,features,delta,_,_,mask=proposal_tensors(batch,evidence,frozen,policy); output=model(features)
        flat=mask.flatten().nonzero().flatten()[::max(config.sample_stride,1)]
        labels.append((delta < -margin).flatten()[flat].cpu().numpy()); probabilities.append(output.harm_probability.flatten()[flat].cpu().numpy())
        gains.append(delta.flatten()[flat].cpu().numpy())
        if output.predicted_gain is not None: predictions.append(output.predicted_gain.flatten()[flat].cpu().numpy())
    label=np.concatenate(labels); probability=np.concatenate(probabilities); gain=np.concatenate(gains)
    result={"harm_auroc":float(roc_auc_score(label,probability)),"harm_auprc":float(average_precision_score(label,probability)),
            "brier":float(brier_score_loss(label,probability)),"sample_count":len(label),"harm_fraction":float(label.mean())}
    if predictions:
        predicted=np.concatenate(predictions); result["gain_mae"]=float(mean_absolute_error(gain,predicted)); result["gain_correlation"]=float(np.corrcoef(gain,predicted)[0,1])
    policies=[]
    gain_thresholds=(-.25,-.10,-.05,0.,.025,.05,.10,.20) if predictions else (-math.inf,)
    for harm_threshold in np.linspace(.05,.95,19):
        for gain_threshold in gain_thresholds:
            accepted=(probability<harm_threshold)
            if predictions: accepted &= predicted>gain_threshold
            if not accepted.any(): continue
            harmful=float((gain[accepted] < -margin).mean()); coverage=float(accepted.mean()); utility=float((gain*accepted).mean())
            policies.append((utility,harmful,coverage,float(harm_threshold),float(gain_threshold)))
    feasible=[item for item in policies if item[1]<=.10 and item[2]>=.01]
    selected=max(feasible or policies,key=lambda item:item[0])
    result.update({"best_selective_gain":selected[0],"best_selective_harm":selected[1],"best_selective_coverage":selected[2],
                   "best_selective_harm_threshold":selected[3],"best_selective_gain_threshold":selected[4],"selective_feasible":bool(feasible)})
    return result


def train_models(config: argparse.Namespace, tiny: str | None = None) -> None:
    seed_all(config.seed); names=MODEL_NAMES if config.model=="all" else (config.model,)
    sequences=(TRAIN[0],) if tiny else TRAIN; original=config.max_frames
    if tiny: config.max_frames=20
    config.configuration="soft"; train_data=build_ram_bank(config,sequences,training=True)
    if tiny == "overfit": validation_data=train_data
    else:
        if tiny: config.max_frames=20
        validation_data=build_ram_bank(config,(VALIDATION[0],) if tiny else VALIDATION,training=False)
    config.max_frames=original; device=torch.device(config.device); frozen,state=load_frozen(device); policy=frozen_policy()
    ages,provenance=state["ages"],state["provenance"]
    for name in names:
        root=model_root(config,name); (root/"checkpoints").mkdir(parents=True,exist_ok=True)
        model=gate_model(name,config,device); optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        epochs=80 if tiny=="overfit" else 3 if tiny=="smoke" else config.epochs
        data=loader(train_data,config,training=True); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(epochs*len(data),1),eta_min=config.learning_rate*.05)
        scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda"); history=[]; best=-math.inf; first=None; started=time.perf_counter()
        for epoch in range(epochs):
            model.train(); data.balanced_sampler.set_epoch(epoch); totals=defaultdict(float); steps=0
            for cpu in data:
                batch=to_device(cpu,device); evidence=evidence_from_batch(batch,ages,provenance)
                proposal,features,delta,_,_,mask=proposal_tensors(batch,evidence,frozen,policy)
                with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
                    output=model(features); loss,parts=training_loss(output,delta,mask,config.margin,name=="joint_gain_risk")
                optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(optimizer); scaler.update(); scheduler.step()
                if first is None: first=float(loss.detach())
                totals["loss"]+=float(loss.detach()); totals["harm_loss"]+=parts["harm"]; totals["delta_loss"]+=parts["delta"]; steps+=1
            diagnostics=sampled_diagnostics(model,validation_data,frozen,state,config,config.margin)
            row={"epoch":epoch+1,**{key:value/max(steps,1) for key,value in totals.items()},**{f"validation_{key}":value for key,value in diagnostics.items()},"learning_rate":scheduler.get_last_lr()[0]}
            history.append(row); score=diagnostics["best_selective_gain"]
            payload={"project":"ARGOS v2","model_name":name,"model":model.state_dict(),"hidden":config.hidden,"depth":config.depth,"joint":name=="joint_gain_risk",
                     "feature_names":GATE_FEATURE_NAMES,"frozen_checkpoint_sha256":EXPECTED_FROZEN_SHA256,"split":split_manifest(),"margin":config.margin,"epoch":epoch+1}
            torch.save(payload,root/"checkpoints/last.pt")
            if score>best: best=score; torch.save(payload,root/"checkpoints/best_validation.pt")
            print(json.dumps(clean({"model":name,**row})),flush=True)
        write_csv(root/"training_history.csv",history)
        best_train_loss=min(row["loss"] for row in history)
        summary={"model":name,"parameters":sum(p.numel() for p in model.parameters()),"initial_loss":first,"final_loss":history[-1]["loss"],"best_train_loss":best_train_loss,"best_validation_selective_gain":best,"wall_time_s":time.perf_counter()-started}
        if tiny:
            # The logistic model is an identifiability baseline, not an
            # over-parameterized memorization test.  Require strong tiny-set
            # fitting only from the nonlinear gates.
            required=.70 if tiny=="overfit" else .95
            best_ap=max(row["validation_harm_auprc"] for row in history)
            summary["best_validation_harm_auprc"]=best_ap
            summary["passed"]=bool(math.isfinite(best_train_loss) and (name=="logistic" or best_train_loss<float(first)*required or best_ap>=.80))
            save_json(root/f"{tiny}_summary.json",summary)
            if not summary["passed"]: raise RuntimeError(f"{name} {tiny} did not reduce loss sufficiently")
        else: save_json(root/"training_summary.json",summary)


def load_gate(config: argparse.Namespace, name: str, device: torch.device):
    path=model_root(config,name)/"checkpoints/best_validation.pt"; state=torch.load(path,map_location="cpu",weights_only=False)
    model=gate_model(name,config,device); model.load_state_dict(state["model"]); model.eval(); return model,state,path


def policy_masks(name: str, arrays: dict[str,np.ndarray], candidate: dict) -> np.ndarray:
    eligible=arrays["eligible"].astype(bool)
    if name=="ungated": return eligible
    if name=="score_threshold": return eligible & (arrays["score"]>=candidate["score_min"])
    if name=="magnitude_threshold": return eligible & (arrays["update"]<=candidate["update_max"])
    if name in ("logistic","harm_mlp"): return eligible & (arrays[name+"_harm"]<candidate["harm_max"])
    if name=="joint_gain_risk": return eligible & (arrays[name+"_harm"]<candidate["harm_max"]) & (arrays[name+"_gain"]>candidate["gain_min"])
    raise KeyError(name)


def policy_metrics(arrays: dict[str,np.ndarray], accepted: np.ndarray, margin: float) -> dict:
    valid=arrays["valid"].astype(bool); accepted=accepted&valid; delta=arrays["delta"]; raw_error=arrays["raw_error"]
    harmful=accepted&(delta < -margin); helpful=accepted&(delta > margin); clean=valid&(raw_error<=margin); clean_deg=accepted&clean&(delta < -margin)
    frame_ids=arrays["frame_group"]; group_count=int(frame_ids.max())+1 if len(frame_ids) else 0
    frame_valid=np.bincount(frame_ids,weights=valid.astype(np.float64),minlength=group_count)
    frame_harm=np.bincount(frame_ids,weights=(-delta*accepted*valid),minlength=group_count)
    frame_delta=np.divide(frame_harm,frame_valid,out=np.zeros_like(frame_harm),where=frame_valid>0)
    gain=float((delta*accepted)[valid].mean()); coverage=float(accepted[valid].mean()); changed=max(int(accepted.sum()),1)
    return {"gain":gain,"output_epe":float(raw_error[valid].mean()-gain),"coverage":coverage,"exact_raw_fallback_frequency":1-coverage,
            "intervention_precision":float(helpful.sum()/changed),"harmful_update_rate":float(harmful.sum()/changed),
            "harmful_update_rate_all_valid":float(harmful.sum()/max(int(valid.sum()),1)),"clean_degradation":float(clean_deg.sum()/max(int(clean.sum()),1)),
            "degraded_frame_fraction":float(np.mean(frame_delta>0)),"worst_frame_degradation":float(frame_delta.max(initial=0.0)),
            "gain_over_fixed_h4":float(FIXED_H4_EPE-(raw_error[valid].mean()-gain)),"valid_count":int(valid.sum()),"accepted_count":int(accepted.sum())}


@torch.inference_mode()
def collect_arrays(config: argparse.Namespace, dataset, gates: dict[str,torch.nn.Module]) -> dict[str,np.ndarray]:
    device=torch.device(config.device); frozen,state=load_frozen(device); policy=frozen_policy(); ages,provenance=state["ages"],state["provenance"]
    values=defaultdict(list); offset=0
    for cpu in loader(dataset,config,training=False):
        batch=to_device(cpu,device); evidence=evidence_from_batch(batch,ages,provenance); proposal,features,delta,raw_error,_,_=proposal_tensors(batch,evidence,frozen,policy)
        base=((batch["gt_coverage"].float()>.50)&batch["raw_valid"].bool()); batch_size=len(batch["raw"])
        score=proposal.top1_score; update=(proposal.proposed-evidence.raw).abs()
        chosen_age=torch.gather(torch.tensor(ages,device=device)[None,:,None,None].expand(batch_size,-1,*evidence.raw.shape[-2:]),1,proposal.chosen)
        tensors={"valid":base,"eligible":proposal.eligible,"delta":delta,"raw_error":raw_error,"score":score,"update":update,"age":chosen_age}
        for name,model in gates.items():
            output=model(features); tensors[name+"_harm"]=output.harm_probability
            if output.predicted_gain is not None: tensors[name+"_gain"]=output.predicted_gain
        stride=max(config.sample_stride,1); pixels=evidence.raw.shape[-2]*evidence.raw.shape[-1]; total=batch_size*pixels
        selected=np.arange(0,total,stride,dtype=np.int64)
        for key,tensor in tensors.items(): values[key].append(tensor.reshape(-1)[selected].cpu().numpy())
        values["frame_group"].append((selected//pixels+offset).astype(np.int32)); offset+=batch_size
    return {key:np.concatenate(items) for key,items in values.items()}


def calibration_candidates(name: str):
    if name=="ungated": return [{}]
    if name=="score_threshold": return [{"score_min":value} for value in (.10,.15,.20,.30,.50,.75,1.0)]
    if name=="magnitude_threshold": return [{"update_max":value} for value in (.01,.025,.05,.10,.20,.50,1.,2.,8.)]
    if name in ("logistic","harm_mlp"): return [{"harm_max":value} for value in np.linspace(.01,.99,99)]
    if name=="joint_gain_risk": return [{"harm_max":harm,"gain_min":gain} for harm in np.linspace(.05,.95,19) for gain in (-.25,-.10,-.05,0,.025,.05,.10,.20,.50)]
    raise KeyError(name)


def calibrate(config: argparse.Namespace) -> None:
    config.configuration="soft"; dataset=build_ram_bank(config,VALIDATION,training=False); device=torch.device(config.device)
    gates={name:load_gate(config,name,device)[0] for name in MODEL_NAMES}; arrays=collect_arrays(config,dataset,gates); rows=[]; policies={}
    for margin in MARGINS:
        for name in ("ungated","score_threshold","magnitude_threshold")+MODEL_NAMES:
            candidates=[]
            for candidate in calibration_candidates(name):
                metrics=policy_metrics(arrays,policy_masks(name,arrays,candidate),margin); row={"policy":name,"margin":margin,**candidate,**metrics}; rows.append(row); candidates.append(row)
            if margin==config.margin:
                feasible=[row for row in candidates if row["harmful_update_rate"]<=.10 and row["clean_degradation"]<=.03 and row["degraded_frame_fraction"]<=.25 and row["coverage"]>=.001]
                selected=max(feasible or candidates,key=lambda row:(row["gain"],row["intervention_precision"])); policies[name]={key:value for key,value in selected.items() if key not in {"output_epe","gain_over_fixed_h4","valid_count","accepted_count"}}
    freeze={"project":"ARGOS v2","selected_only_on":list(VALIDATION),"dataset7_accessed":False,"frozen_checkpoint_sha256":EXPECTED_FROZEN_SHA256,
            "margin":config.margin,"constraints":{"harmful_update_rate":.10,"clean_degradation":.03,"degraded_frame_fraction":.25,"minimum_coverage":.001},"policies":policies}
    write_csv(config.output/"validation_summary.csv",rows); save_json(config.output/"calibration/frozen_policy.json",freeze)
    save_json(config.output/"calibration/freeze_manifest.json",freeze|{"gate_checkpoint_hashes":{name:sha256(model_root(config,name)/"checkpoints/best_validation.pt") for name in MODEL_NAMES}})


def _frame_row(method, raw, output, fixed_h4, age1, gt, gt_memory, gt_memory_valid, valid, accepted, chosen_age, metadata):
    raw_error=(raw-gt).abs(); error=(output-gt).abs(); changed=accepted&valid; delta=raw_error-error; helpful=changed&(delta>.10); harmful=changed&(delta<-.10); clean=valid&(raw_error<=.10)
    temporal=valid&gt_memory_valid.bool(); gt_delta=gt-gt_memory; raw_temporal=((raw-age1)-gt_delta).abs(); output_temporal=((output-age1)-gt_delta).abs()
    return {"method":method,**metadata,"valid_count":int(valid.sum()),"raw_error_sum":float(raw_error[valid].sum()),"output_error_sum":float(error[valid].sum()),
            "fixed_h4_error_sum":float((fixed_h4-gt).abs()[valid].sum()),"changed_count":int(changed.sum()),"helpful_count":int(helpful.sum()),"harmful_count":int(harmful.sum()),"clean_count":int(clean.sum()),
            "clean_degradation_count":int((changed&clean&(delta<-.10)).sum()),"frame_raw_epe":float(raw_error[valid].mean()),"frame_output_epe":float(error[valid].mean()),
            "frame_degradation":float(error[valid].mean()-raw_error[valid].mean()),"exact_fallback_count":int((valid&~changed).sum()),
            "temporal_count":int(temporal.sum()),"raw_tepe_sum":float(raw_temporal[temporal].sum()),"output_tepe_sum":float(output_temporal[temporal].sum()),
            "raw_teper_sum":float((raw_temporal/(gt_delta.abs()+1e-3))[temporal].sum()),"output_teper_sum":float((output_temporal/(gt_delta.abs()+1e-3))[temporal].sum()),
            **{f"selected_age_{age}":int((changed&(chosen_age==age)).sum()) for age in RAW_ANCHOR_AGES}}


@torch.inference_mode()
def evaluate(config: argparse.Namespace) -> None:
    freeze_path=config.output/"calibration/freeze_manifest.json"
    if config.split=="test" and not freeze_path.exists(): raise RuntimeError("dataset 7 locked until complete validation policy freeze")
    freeze=json.loads(freeze_path.read_text()); sequences=TEST if config.split=="test" else VALIDATION
    config.configuration="soft"; dataset=build_ram_bank(config,sequences,training=False); device=torch.device(config.device); frozen,state=load_frozen(device); base_policy=frozen_policy()
    gates={name:load_gate(config,name,device)[0] for name in MODEL_NAMES}; ages,provenance=state["ages"],state["provenance"]
    methods=("ungated","score_threshold","magnitude_threshold")+MODEL_NAMES; rows=[]; diagnostics=defaultdict(list); offset=0
    for cpu in loader(dataset,config,training=False):
        batch=to_device(cpu,device); evidence=evidence_from_batch(batch,ages,provenance); proposal,features,delta,raw_error,_,_=proposal_tensors(batch,evidence,frozen,base_policy); batch_size=len(batch["raw"])
        tensor_arrays={"eligible":proposal.eligible,"score":proposal.top1_score,"update":abs(proposal.proposed-evidence.raw)}; outputs={}
        for name,model in gates.items():
            outputs[name]=model(features); tensor_arrays[name+"_harm"]=outputs[name].harm_probability
            if outputs[name].predicted_gain is not None: tensor_arrays[name+"_gain"]=outputs[name].predicted_gain
        age_map=torch.tensor(ages,device=device)[None,:,None,None].expand(batch_size,-1,*evidence.raw.shape[-2:]); chosen_age=torch.gather(age_map,1,proposal.chosen)
        for method in methods:
            candidate=freeze["policies"][method]; arrays={key:value.cpu().numpy() for key,value in tensor_arrays.items()}; gate_accept=torch.from_numpy(policy_masks(method,arrays,candidate)).to(device)
            prediction,accepted=apply_veto(evidence.raw,proposal,gate_accept); valid=(batch["gt_coverage"].float()>.50)&batch["raw_valid"].bool()
            for local in range(batch_size): rows.append(_frame_row(method,evidence.raw[local:local+1],prediction[local:local+1],batch["fixed_h4_output"][local:local+1].float(),evidence.candidates[local:local+1,0:1],batch["gt"][local:local+1].float(),batch["gt_memory"][local:local+1].float(),batch["gt_memory_valid"][local:local+1].bool(),valid[local:local+1],accepted[local:local+1],chosen_age[local:local+1],dataset.metadata[offset+local]))
            diagnostics[method+"_accepted"].append(accepted.flatten().cpu().numpy()); diagnostics[method+"_delta"].append(delta.flatten().cpu().numpy()); diagnostics[method+"_age"].append(chosen_age.flatten().cpu().numpy())
        diagnostics["valid"].append(((batch["gt_coverage"].float()>.50)&batch["raw_valid"].bool()).flatten().cpu().numpy())
        diagnostics["raw_error"].append(raw_error.flatten().cpu().numpy())
        pixels=evidence.raw.shape[-2]*evidence.raw.shape[-1]
        diagnostics["frame_group"].append(np.repeat(np.arange(offset,offset+batch_size,dtype=np.int32),pixels))
        diagnostics["eligible"].append(proposal.eligible.flatten().cpu().numpy()); diagnostics["update"].append((proposal.proposed-evidence.raw).abs().flatten().cpu().numpy())
        diagnostics["agreement"].append((features[:,12:13]*len(ages)).round().flatten().cpu().numpy())
        for name,output in outputs.items():
            mask=proposal.eligible & ((batch["gt_coverage"].float()>.50)&batch["raw_valid"].bool())
            diagnostics[name+"_label"].append((delta<-.10)[mask].cpu().numpy()); diagnostics[name+"_prob"].append(output.harm_probability[mask].cpu().numpy()); diagnostics[name+"_true_gain"].append(delta[mask].cpu().numpy())
            if output.predicted_gain is not None: diagnostics[name+"_pred_gain"].append(output.predicted_gain[mask].cpu().numpy())
        offset+=batch_size
    destination=config.output/("frozen_test" if config.split=="test" else "validation"); destination.mkdir(parents=True,exist_ok=True); write_csv(destination/"frame_metrics.csv",rows)
    summaries=[]; per_backbone=[]; per_sequence=[]; per_age=[]
    for method in methods:
        method_rows=[row for row in rows if row["method"]==method]; count=sum(row["valid_count"] for row in method_rows); raw_sum=sum(row["raw_error_sum"] for row in method_rows); out_sum=sum(row["output_error_sum"] for row in method_rows); changed=sum(row["changed_count"] for row in method_rows); harmful=sum(row["harmful_count"] for row in method_rows); helpful=sum(row["helpful_count"] for row in method_rows); clean=sum(row["clean_count"] for row in method_rows); clean_deg=sum(row["clean_degradation_count"] for row in method_rows); frame=np.asarray([row["frame_degradation"] for row in method_rows],dtype=np.float64); finite_frame=frame[np.isfinite(frame)]
        fixed=sum(row["fixed_h4_error_sum"] for row in method_rows); temporal=sum(row["temporal_count"] for row in method_rows)
        accepted_array=np.concatenate(diagnostics[method+"_accepted"]); update_array=np.concatenate(diagnostics["update"])
        summary={"policy":method,"split":config.split,"valid_count":count,"raw_epe":raw_sum/count,"output_epe":out_sum/count,"gain":(raw_sum-out_sum)/count,"fixed_h4_epe_common_support":fixed/count,"gain_over_fixed_h4":(fixed-out_sum)/count,"raw_bank_oracle_recovery":((raw_sum-out_sum)/count)/RAW_BANK_ORACLE_GAIN,
                 "coverage":changed/count,"exact_raw_fallback_frequency":1-changed/count,"intervention_precision":helpful/max(changed,1),"harmful_update_rate":harmful/max(changed,1),"harmful_update_rate_all_valid":harmful/count,
                 "clean_degradation":clean_deg/max(clean,1),"degraded_frame_fraction":float((finite_frame>0).mean()) if finite_frame.size else 0.0,"worst_frame_degradation":float(finite_frame.max()) if finite_frame.size else None,"ungated_gain_retained":(raw_sum-out_sum)/max(sum(r["raw_error_sum"]-r["output_error_sum"] for r in rows if r["method"]=="ungated"),1e-12),
                 "mean_accepted_update":float(update_array[accepted_array].mean()) if accepted_array.any() else 0.,
                 "raw_tepe":sum(r["raw_tepe_sum"] for r in method_rows)/max(temporal,1),"output_tepe":sum(r["output_tepe_sum"] for r in method_rows)/max(temporal,1),"raw_teper":sum(r["raw_teper_sum"] for r in method_rows)/max(temporal,1),"output_teper":sum(r["output_teper_sum"] for r in method_rows)/max(temporal,1)}
        summaries.append(summary)
        for row in grouped_simple(method_rows,"backbone"): per_backbone.append(row)
        for row in grouped_simple(method_rows,"sequence"): per_sequence.append(row)
        accepted=np.concatenate(diagnostics[method+"_accepted"]); age=np.concatenate(diagnostics[method+"_age"]); valid=np.concatenate(diagnostics["valid"])
        for value in RAW_ANCHOR_AGES: per_age.append({"policy":method,"age":value,"accepted_count":int((accepted&valid&(age==value)).sum()),"accepted_fraction":float((accepted&valid&(age==value)).sum()/max((accepted&valid).sum(),1))})
    write_csv(destination/"summary.csv",summaries); write_csv(destination/"per_backbone_metrics.csv",per_backbone); write_csv(destination/"per_sequence_metrics.csv",per_sequence); write_csv(destination/"per_age_metrics.csv",per_age)
    write_csv(destination/"temporal_metrics.csv",[{key:row[key] for key in ("policy","raw_tepe","output_tepe","raw_teper","output_teper")} for row in summaries])
    detection=[]
    for name in MODEL_NAMES:
        label=np.concatenate(diagnostics[name+"_label"]); probability=np.concatenate(diagnostics[name+"_prob"]); row={"model":name,"harm_auroc":roc_auc_score(label,probability),"harm_auprc":average_precision_score(label,probability),"brier":brier_score_loss(label,probability),"ece":ece(label,probability)}
        if name+"_pred_gain" in diagnostics: row["gain_mae"]=mean_absolute_error(np.concatenate(diagnostics[name+"_true_gain"]),np.concatenate(diagnostics[name+"_pred_gain"]))
        detection.append(row)
    valid=np.concatenate(diagnostics["valid"]); eligible=np.concatenate(diagnostics["eligible"]); delta=np.concatenate(diagnostics["ungated_delta"]); raw_error=np.concatenate(diagnostics["raw_error"]); frame_group=np.concatenate(diagnostics["frame_group"])
    oracle_arrays={"valid":valid,"eligible":eligible,"delta":delta,"raw_error":raw_error,"frame_group":frame_group}
    oracle_rows=[]
    for label,accepted in (("oracle_reject_harm_gt_margin",eligible&(delta>=-.10)),("oracle_positive_gain_only",eligible&(delta>0)),("oracle_help_margin_only",eligible&(delta>.10))):
        oracle_rows.append({"oracle":label,**policy_metrics(oracle_arrays,accepted,.10)})
    write_csv(destination/"oracle_rejection_ceiling.csv",oracle_rows)
    update=np.concatenate(diagnostics["update"]); agreement=np.concatenate(diagnostics["agreement"]); analysis=[]
    for label,lower,upper in (("lt001",0,.01),("001_005",.01,.05),("005_010",.05,.10),("010_050",.10,.50),("ge050",.50,float("inf"))):
        mask=valid&(update>=lower)&(update<upper); analysis.append({"bin":label,"valid_count":int(mask.sum()),"mean_delta":float(delta[mask].mean()) if mask.any() else 0.,"harmful_fraction":float((delta[mask]<-.10).mean()) if mask.any() else 0.})
    write_csv(destination/"update_magnitude_analysis.csv",analysis)
    agreement_rows=[]
    for count_value in range(1,len(ages)+1):
        mask=valid&(agreement==count_value); agreement_rows.append({"agreement_count":count_value,"valid_count":int(mask.sum()),"mean_delta":float(delta[mask].mean()) if mask.any() else 0.,"harmful_fraction":float((delta[mask]<-.10).mean()) if mask.any() else 0.})
    write_csv(destination/"agreement_analysis.csv",agreement_rows)
    write_csv(destination/"harm_detection_metrics.csv",detection); save_json(destination/"summary.json",{"project":"ARGOS v2","split":config.split,"policies":summaries,"harm_detection":detection,"oracle_ceilings":oracle_rows})
    if config.split=="test": save_json(config.output/"protocol_audit/test_opened.json",{"opened_after_freeze":True,"freeze_manifest_sha256":sha256(freeze_path),"sequences":list(TEST)})


def grouped_simple(rows,key):
    groups=defaultdict(list)
    for row in rows: groups[row[key]].append(row)
    result=[]
    for value,items in groups.items():
        count=sum(r["valid_count"] for r in items); raw=sum(r["raw_error_sum"] for r in items); out=sum(r["output_error_sum"] for r in items); changed=sum(r["changed_count"] for r in items); harmful=sum(r["harmful_count"] for r in items)
        result.append({"policy":items[0]["method"],key:value,"raw_epe":raw/count,"output_epe":out/count,"gain":(raw-out)/count,"coverage":changed/count,"harmful_update_rate":harmful/max(changed,1)})
    return result


def ece(label, probability, bins=15):
    edges=np.linspace(0,1,bins+1); value=0.0
    for left,right in zip(edges[:-1],edges[1:]):
        mask=(probability>=left)&(probability<(right if right<1 else right+1e-9))
        if mask.any(): value+=mask.mean()*abs(label[mask].mean()-probability[mask].mean())
    return float(value)


def audit(config):
    config.output.mkdir(parents=True,exist_ok=True); manifest=split_manifest()|{"project":"ARGOS v2","experiment":"post-hoc veto-only selective gate","frozen_checkpoint":str(FROZEN_CHECKPOINT),"frozen_checkpoint_sha256":sha256(FROZEN_CHECKPOINT),"expected_sha256":EXPECTED_FROZEN_SHA256,"dataset7_locked_until_freeze":True,"gate_can_open":False,"gate_can_change_candidate":False,"gate_output_enters_anchor_bank":False,"gt_in_inference_features":False,"feature_count":GATE_FEATURE_CHANNELS}
    save_json(config.output/"protocol_audit/protocol_audit.json",manifest); save_json(config.output/"frozen_checkpoint_hashes.json",{"raw_multi_anchor":sha256(FROZEN_CHECKPOINT),"frozen_policy":sha256(FROZEN_POLICY)})
    save_json(config.output/"feature_schema.json",{"features":list(GATE_FEATURE_NAMES),"normalization":"fixed physical clipping; no dataset-7 statistics","contains_gt":False,"contains_backbone_identity":False})


def summarize(config):
    test=json.loads((config.output/"frozen_test/summary.json").read_text()); validation=json.loads((config.output/"validation/summary.json").read_text()) if (config.output/"validation/summary.json").exists() else None
    policies={row["policy"]:row for row in test["policies"]}
    frozen=json.loads((config.output/"calibration/frozen_policy.json").read_text())
    selected_name="joint_gain_risk"
    # The primary policy is selected on validation and must never be replaced by
    # the test-set EPE winner.  Keeping this explicit prevents test leakage.
    if selected_name not in policies:
        raise RuntimeError(f"Frozen validation policy {selected_name!r} missing from test evaluation")
    # Recompute frame-level tail statistics from the immutable per-frame report.
    # Older summaries could contain null tails when an empty frame entered the
    # reduction; this is reporting-only and does not reopen or retune dataset 7.
    frame_report=list(csv.DictReader((config.output/"frozen_test/frame_metrics.csv").open()))
    for name,row in policies.items():
        vals=[]
        for item in frame_report:
            if item.get("method")!=name: continue
            try:
                value=float(item.get("frame_degradation","nan"))
            except (TypeError,ValueError):
                value=float("nan")
            if np.isfinite(value): vals.append(value)
        if vals:
            row["worst_frame_degradation"]=float(np.max(vals))
            row["degraded_frame_fraction"]=float(np.mean(np.asarray(vals)>0))
    best=policies[selected_name]
    gates={"beats_fixed_h4":best["gain_over_fixed_h4"]>0,"harm_le_10pct":best["harmful_update_rate"]<=.10,"clean_le_3pct":best["clean_degradation"]<=.03,"degraded_frames_le_25pct":best["degraded_frame_fraction"]<=.25,"nontrivial_coverage":best["coverage"]>=.001}
    back=list(csv.DictReader((config.output/"frozen_test/per_backbone_metrics.csv").open())); seq=list(csv.DictReader((config.output/"frozen_test/per_sequence_metrics.csv").open()))
    gates["all_seen_backbones_positive"]=all(float(row["gain"])>0 for row in back if row["policy"]==best["policy"]); gates["nearly_all_sequences_positive"]=np.mean([float(row["gain"])>0 for row in seq if row["policy"]==best["policy"]])>=.75
    verdict="GO" if all(gates.values()) else "CONDITIONAL GO" if best["gain"]>0 and best["harmful_update_rate"]<=.10 and best["clean_degradation"]<=.03 else "NO-GO"
    aggregate={"project":"ARGOS v2","frozen_ungated":policies["ungated"],"selected_validation_policy":selected_name,"best_validation_selected_test_policy":best,"policies":test["policies"],"harm_detection":test["harm_detection"],"promotion_gates":gates,"geometry_verdict":"GO" if best["gain"]>0 else "NO-GO","safety_verdict":"GO" if best["harmful_update_rate"]<=.10 and best["clean_degradation"]<=.03 else "NO-GO","verdict":verdict,"scope":"held-out SCARED-C dataset 7; three training-seen backbones; no unseen/OOD claim"}
    save_json(config.output/"aggregate_summary.json",aggregate); save_json(config.output/"verdicts.json",{"verdict":verdict,"gates":gates})
    for source,target in (("frozen_test/summary.csv","frozen_test_summary.csv"),("frozen_test/per_backbone_metrics.csv","per_backbone_metrics.csv"),("frozen_test/per_sequence_metrics.csv","per_sequence_metrics.csv"),("frozen_test/per_age_metrics.csv","per_age_metrics.csv"),("frozen_test/harm_detection_metrics.csv","harm_detection_metrics.csv"),("validation_summary.csv","risk_coverage.csv")):
        path=config.output/source
        if path.exists(): (config.output/target).write_bytes(path.read_bytes())
    for filename in ("temporal_metrics.csv","update_magnitude_analysis.csv","agreement_analysis.csv","oracle_rejection_ceiling.csv"):
        path=config.output/"frozen_test"/filename
        if path.exists(): (config.output/filename).write_bytes(path.read_bytes())
    validation_rows=list(csv.DictReader((config.output/"validation_summary.csv").open())); fixed=[]
    for policy in sorted({row["policy"] for row in validation_rows}):
        candidates=[row for row in validation_rows if row["policy"]==policy and abs(float(row["margin"])-.10)<1e-9]
        for target in (.05,.10,.15):
            feasible=[row for row in candidates if float(row["harmful_update_rate"])<=target and float(row["coverage"])>=.001]
            if feasible: fixed.append({"harm_target":target,**max(feasible,key=lambda row:float(row["gain"]))})
    write_csv(config.output/"fixed_risk_operating_points.csv",fixed)
    training=[]
    for name in MODEL_NAMES:
        path=model_root(config,name)/"training_history.csv"
        if path.exists(): training.extend({"model":name,**row} for row in csv.DictReader(path.open()))
    write_csv(config.output/"training_summary.csv",training)
    write_csv(config.output/"ablation_summary.csv",test["policies"]); write_csv(config.output/"calibration_metrics.csv",test["harm_detection"]); write_csv(config.output/"gain_regression_metrics.csv",[row for row in test["harm_detection"] if "gain_mae" in row]); write_csv(config.output/"intervention_metrics.csv",test["policies"])
    (config.output/"paper_ready_tables.tex").write_text("\\begin{tabular}{lrrrrr}\\toprule\nPolicy & EPE & Gain & Coverage & Harm & Clean degradation \\\\\n"+"\n".join(f"{r['policy'].replace('_',' ')} & {r['output_epe']:.4f} & {r['gain']:.4f} & {r['coverage']:.3f} & {r['harmful_update_rate']:.3f} & {r['clean_degradation']:.3f} \\\\" for r in test["policies"])+"\n\\bottomrule\\end{tabular}\n")
    (config.output/"README.md").write_text(f"# ARGOS v2 raw multi-anchor selective gate\n\nFrozen dataset-7 verdict: **{verdict}**. Best validation-selected policy: `{best['policy']}`; EPE {best['output_epe']:.5f}, gain {best['gain']:.5f}, gain over fixed H=4 {best['gain_over_fixed_h4']:.5f}, accepted harm {100*best['harmful_update_rate']:.2f}%, clean degradation {100*best['clean_degradation']:.2f}%, coverage {100*best['coverage']:.2f}%.\n\nThis is a veto-only post-hoc gate over a frozen proposal. Unseen-backbone and external-dataset OOD remain unverified. No clinical-safety or formal conformal claim is made.\n")


def main():
    config=arguments(); audit(config)
    if config.mode=="audit": return
    if config.mode=="overfit": train_models(config,"overfit")
    elif config.mode=="smoke": train_models(config,"smoke")
    elif config.mode=="train": train_models(config)
    elif config.mode=="calibrate": calibrate(config)
    elif config.mode=="evaluate": evaluate(config)
    elif config.mode=="summarize": summarize(config)


if __name__ == "__main__": main()
