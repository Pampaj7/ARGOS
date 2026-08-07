#!/usr/bin/env python3
"""Train one frozen-architecture ARGOS v2 budget/seed run."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import socket
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from campaign_common import *
from data_pipeline import build_ram_bank, evidence_from_batch, make_loader, to_device
from losses import MultiAnchorLossConfig, multi_anchor_targets, raw_multi_anchor_losses
from argos_freezed.models.raw_multi_anchor_refiner import RawMultiAnchorRefiner


def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def validation_epoch(model, dataset, *, seed, batch_size, workers, device, loss_config):
    model.eval(); totals = defaultdict(float); batches = 0
    probability_values=[]; score_values=[]; raw_values=[]; candidate_values=[]; weight_values=[]; gt_values=[]
    for cpu in make_loader(dataset, seed=seed, batch_size=batch_size, workers=workers, training=False):
        batch=to_device(cpu,device); evidence=evidence_from_batch(batch)
        targets=multi_anchor_targets(batch["raw"].float(),evidence.candidates,batch["gt"].float(),batch["gt_coverage"].float(),batch["raw_valid"].bool(),evidence,margin_px=.1)
        output=model(evidence); losses=raw_multi_anchor_losses(output,evidence,targets,loss_config,enable_fusion=True)
        for key,value in losses.items(): totals[key]+=float(value)
        score,chosen=output.selection_score.max(dim=1,keepdim=True); probability=torch.gather(output.utility_probability,1,chosen)
        candidate=torch.gather(evidence.candidates,1,chosen); weight=torch.gather(output.fusion_weight,1,chosen)
        base=((batch["gt_coverage"].float()>.50)&batch["raw_valid"].bool()).flatten(); index=base.nonzero().flatten()[::64]
        take=lambda value:value.flatten()[index].float().cpu().numpy()
        probability_values.append(take(probability)); score_values.append(take(score)); raw_values.append(take(evidence.raw))
        candidate_values.append(take(candidate)); weight_values.append(take(weight)); gt_values.append(take(batch["gt"].float())); batches+=1
    probability,score,raw,candidate,weight,gt=[np.concatenate(value) for value in (probability_values,score_values,raw_values,candidate_values,weight_values,gt_values)]
    true_gain=np.abs(raw-gt)-np.abs(raw+weight*(candidate-raw)-gt); policies=[]
    for p in (.3,.4,.5,.6,.7,.8,.9):
        for u in (-.05,0,.01,.02,.05,.1):
            accepted=(probability>=p)&(score>=u); realized=true_gain*accepted
            positive=float(realized[realized>0].sum()); negative=float((-realized[realized<0]).sum())
            policies.append((float(realized.mean()),float(accepted.mean()),negative/max(positive,1e-12),p,u))
    feasible=[item for item in policies if item[1]>=.005 and item[2]<=.25]; selected=max(feasible or policies,key=lambda item:item[0])
    return {**{f"validation_{key}":value/max(batches,1) for key,value in totals.items()},"validation_best_gain":selected[0],
            "validation_best_coverage":selected[1],"validation_harm_cost_fraction":selected[2],"validation_probability_threshold":selected[3],
            "validation_utility_threshold":selected[4],"validation_constraint_feasible":bool(feasible)}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--budget",type=int,choices=(1,3,6),required=True); parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    parser.add_argument("--device",default="cuda:0"); parser.add_argument("--workers",type=int,default=48); parser.add_argument("--flow-batch-size",type=int,default=32)
    parser.add_argument("--batch-size",type=int,default=12); parser.add_argument("--output",type=Path); parser.add_argument("--max-frames",type=int)
    parser.add_argument("--stop-after-epochs",type=int); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args(); verify_frozen_core(); verify_training_dependencies(); guard_no_d7(TRAIN_SEQUENCES,VALIDATION_SEQUENCES)
    output=(args.output or run_directory(args.budget,args.seed)).resolve(); output.relative_to((ROOT/"experiments").resolve())
    if args.batch_size != 12: raise RuntimeError("canonical effective batch size is 12")
    if not args.smoke and (args.max_frames is not None or args.flow_batch_size != 32 or output != run_directory(args.budget,args.seed)):
        raise RuntimeError("full runs require canonical data, flow batch 32, and registered output directory")
    if args.smoke: output.relative_to((CAMPAIGN/"smoke_tmp").resolve())
    output.mkdir(parents=True,exist_ok=True); (output/"logs").mkdir(exist_ok=True); (output/"checkpoints").mkdir(exist_ok=True)
    epochs=BUDGET_EPOCHS[args.budget]; stop=min(args.stop_after_epochs or epochs,epochs); device=torch.device(args.device)
    seed_all(args.seed); started=time.perf_counter(); started_iso=datetime.now(timezone.utc).isoformat()
    config={"project":"ARGOS v2","experiment_id":"training_budget_scaling","budget":f"{args.budget}x","seed":args.seed,"epochs":epochs,
            "batch_size":args.batch_size,"effective_batch_size":args.batch_size,"workers":args.workers,"flow_batch_size":args.flow_batch_size,
            "learning_rate":.002,"weight_decay":.0001,"scheduler":"CosineAnnealingLR","eta_min":.0001,"warmup":None,"amp":device.type=="cuda",
            "gradient_clip_norm":5.0,"train_ids":[1,3,6],"validation_ids":[2],"test_locked":True,"max_frames":args.max_frames,
            "frozen_manifest_sha256":FROZEN_MANIFEST_SHA}
    import yaml
    (output/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False)); (output/"README.md").write_text(f"# ARGOS v2 {args.budget}x seed {args.seed}\n\nFrozen geometry_v1; only optimization budget varies. Dataset 7 locked.\n")
    state_path=CAMPAIGN/f"initial_weights/seed_{args.seed}.pt"; initial=torch.load(state_path,map_location="cpu",weights_only=False)
    train_sequences=TRAIN_SEQUENCES[:1] if args.smoke else TRAIN_SEQUENCES; validation_sequences=VALIDATION_SEQUENCES[:1] if args.smoke else VALIDATION_SEQUENCES
    train_data, train_flow_s=build_ram_bank(train_sequences,training=True,seed=args.seed,device=device,flow_batch_size=args.flow_batch_size,max_frames=args.max_frames,progress=lambda x:print(x,flush=True))
    validation_data, validation_flow_s=build_ram_bank(validation_sequences,training=False,seed=args.seed,device=device,flow_batch_size=args.flow_batch_size,max_frames=args.max_frames,progress=lambda x:print(x,flush=True))
    train_loader=make_loader(train_data,seed=args.seed,batch_size=args.batch_size,workers=args.workers,training=True); steps_per_epoch=len(train_loader)
    if not args.smoke and steps_per_epoch != 3973: raise RuntimeError(f"canonical steps/epoch changed: {steps_per_epoch}")
    model=RawMultiAnchorRefiner(32,3).to(device); model.load_state_dict(initial["model"],strict=True)
    if tensor_state_sha256(model.state_dict()) != initial["tensor_sha256"]: raise RuntimeError("initial weight parity failure")
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    total_steps=epochs*steps_per_epoch; scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(total_steps,1),eta_min=.0001)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda"); loss_config=MultiAnchorLossConfig(margin_px=.1)
    history=[]; best=-math.inf; start_epoch=0; optimizer_steps=0; final_path=output/"checkpoints/final.pt"
    if args.resume and final_path.exists():
        saved=torch.load(final_path,map_location="cpu",weights_only=False); model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        if (saved["seed"],saved["budget"],saved["total_steps"],saved["steps_per_epoch"],saved["initial_weight_hash"],saved["recipe_hashes"]) != (args.seed,args.budget,total_steps,steps_per_epoch,initial["tensor_sha256"],recipe_hashes()):
            raise RuntimeError("resume checkpoint does not match frozen run identity")
        scheduler.load_state_dict(saved["scheduler"]); scaler.load_state_dict(saved["scaler"]); history=saved["history"]; best=saved["best_score"]
        start_epoch=saved["epoch"]; optimizer_steps=saved["optimizer_steps"]; print(f"RESUME epoch={start_epoch} step={optimizer_steps}",flush=True)
    torch.cuda.reset_peak_memory_stats(device) if device.type=="cuda" else None
    for epoch in range(start_epoch,stop):
        model.train(); train_loader.balanced_sampler.set_epoch(epoch); totals=defaultdict(float); batches=0; wait_s=0.0; iteration_end=time.perf_counter()
        for cpu in train_loader:
            wait_s += time.perf_counter()-iteration_end; batch=to_device(cpu,device); evidence=evidence_from_batch(batch)
            targets=multi_anchor_targets(batch["raw"].float(),evidence.candidates,batch["gt"].float(),batch["gt_coverage"].float(),batch["raw_valid"].bool(),evidence,margin_px=.1)
            with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
                result=model(evidence); losses=raw_multi_anchor_losses(result,evidence,targets,loss_config,enable_fusion=True)
            optimizer.zero_grad(set_to_none=True); scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer_steps+=1
            for key,value in losses.items(): totals[key]+=float(value.detach())
            batches+=1; iteration_end=time.perf_counter()
        row={"epoch":epoch+1,"optimizer_step":optimizer_steps,**{key:value/max(batches,1) for key,value in totals.items()},"learning_rate":scheduler.get_last_lr()[0],"loader_wait_s":wait_s}
        tick=time.perf_counter(); row.update(validation_epoch(model,validation_data,seed=args.seed,batch_size=args.batch_size,workers=args.workers,device=device,loss_config=loss_config)); row["validation_wall_s"]=time.perf_counter()-tick
        history.append(row); score=row["validation_best_gain"]
        payload={"project":"ARGOS v2","model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
                 "epoch":epoch+1,"optimizer_steps":optimizer_steps,"configuration":"soft","ages":list(ANCHOR_AGES),"provenance":[0.,0.,0.,0.],"channels":32,"blocks":3,
                 "loss":asdict(loss_config),"seed":args.seed,"budget":args.budget,"total_steps":total_steps,"steps_per_epoch":steps_per_epoch,"best_score":max(best,score),"history":history}
        payload.update({"initial_weight_hash":initial["tensor_sha256"],"recipe_hashes":recipe_hashes()})
        atomic_torch_save(final_path,payload)
        if score>best: best=score; atomic_torch_save(output/"checkpoints/best_validation.pt",payload)
        write_csv(output/"train_metrics.csv",[{key:value for key,value in item.items() if not key.startswith("validation_")} for item in history]); write_csv(output/"validation_metrics.csv",history)
        state={"status":"completed" if epoch+1==epochs else "running","budget":f"{args.budget}x","seed":args.seed,"epoch":epoch+1,"optimizer_step":optimizer_steps,
               "current_train_loss":row["total"],"latest_d2_gain":row["validation_best_gain"],"best_d2_gain":best,"best_checkpoint":str(output/"checkpoints/best_validation.pt"),"runtime_s":time.perf_counter()-started}
        atomic_json(output/"state.json",state); print(json.dumps(row),flush=True)
    complete=stop==epochs; wall=time.perf_counter()-started
    runtime={"wall_seconds":wall,"flow_build_train_seconds":train_flow_s,"flow_build_validation_seconds":validation_flow_s,"optimizer_steps":optimizer_steps,
             "steps_per_second":optimizer_steps/max(wall,1e-9),"samples_per_second":optimizer_steps*args.batch_size/max(wall,1e-9),
             "peak_vram_mb":torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else 0,"peak_host_ram_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             "workers":args.workers,"host":socket.gethostname(),"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu","complete":complete}
    atomic_json(output/"runtime_summary.json",runtime)
    manifest={"project":"ARGOS v2","experiment_id":"training_budget_scaling","budget":f"{args.budget}x","seed":args.seed,"frozen_manifest_sha":FROZEN_MANIFEST_SHA,
              "source_dependency_hashes":recipe_hashes(),"architecture_hash":sha256(ROOT/"src/argos_freezed/models/raw_multi_anchor_refiner.py"),"initial_weight_hash":initial["tensor_sha256"],
              "best_checkpoint_hash":sha256(output/"checkpoints/best_validation.pt"),"final_checkpoint_hash":sha256(final_path),"optimizer":"AdamW","scheduler":"CosineAnnealingLR",
              "effective_batch_size":args.batch_size,"optimizer_steps":optimizer_steps,"planned_optimizer_steps":total_steps,"epochs":stop,"planned_epochs":epochs,
              "validation_metric":"validation_best_gain","best_epoch":max(history,key=lambda x:x["validation_best_gain"])["epoch"],"best_step":max(history,key=lambda x:x["validation_best_gain"])["optimizer_step"],
              "hardware":runtime,"launch_command":" ".join(sys.argv),"started_at":started_iso,"ended_at":datetime.now(timezone.utc).isoformat(),"exit_status":"complete" if complete else "paused_for_smoke_or_resume"}
    atomic_json(output/"manifest.json",manifest)


if __name__=="__main__": main()
