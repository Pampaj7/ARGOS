#!/usr/bin/env python3
"""D2-only raw/H4/multi strict-common-support evaluator; never opens D7."""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent; sys.path[:0]=[str(HERE),str(Path("/dtu/p1/leopam/ARGOS/ARGOS_FREEZED/src"))]
from campaign_common import *
from data_pipeline import (_align, infer_age_flows, load_frame_gt, load_sequence_cache, load_sequence_info,
                           read_rgb, resize_gt_to_cache_masked, rgb_tensor)
from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
from argos_freezed.models.raw_multi_anchor_refiner import MultiAnchorEvidence, RawMultiAnchorRefiner, retrieve_and_fuse
from provenance.codd_bounded_memory import BoundedMemoryPolicy, advance_state_age
from provenance.codd_style_fusion import CODDStyleFusionHead, FrozenResNet18Layer1
from provenance.h4_infer import infer as h4_infer

OUT=CAMPAIGN/"selection/d2_common_support_frame_metrics.csv"
CANONICAL=ROOT/"checkpoints/raw_multi_anchor_best_validation.pt"
H4=ROOT/"baselines/bounded_h4/MANIFEST.json"
POLICY=HERE/"provenance/canonical_geometry_v1_policy.json"

def policy(state):
    best=max(state["history"],key=lambda row:row["validation_best_gain"])
    return best["validation_probability_threshold"],best["validation_utility_threshold"]

def model(path, device):
    state=torch.load(path,map_location="cpu",weights_only=False); value=RawMultiAnchorRefiner(state["channels"],state["blocks"]).to(device)
    value.load_state_dict(state["model"],strict=True); value.eval(); value.requires_grad_(False); return value,state

def registered_models(device):
    values={}
    for budget in (1,3,6):
        for seed in SEEDS:
            path=run_directory(budget,seed)/"checkpoints/best_validation.pt"; value,state=model(path,device)
            values[(f"{budget}x",seed)]={"model":value,"policy":policy(state),"path":path,"sha256":sha256(path)}
    value,state=model(CANONICAL,device); frozen=json.loads(POLICY.read_text())["policy"]
    values[("canonical",0)]={"model":value,"policy":(frozen["probability_threshold"],frozen["utility_threshold_px"]),"path":CANONICAL,"sha256":sha256(CANONICAL)}
    return values

def h4_outputs(raw, valid, gt, coverage, images, right, flows, h4, extractor, device):
    state=None; age=0; accumulated=0.; output={}; support={}; reset=BoundedMemoryPolicy(name="fixed_h4",max_age=4)
    for index in range(1,len(raw)):
        item={"raw":torch.from_numpy(raw[index:index+1,None]).float().to(device),"raw_valid":torch.from_numpy(valid[index:index+1,None]).bool().to(device),
              "gt":torch.from_numpy(gt[index:index+1,None]).float().to(device),"gt_coverage":torch.from_numpy(coverage[index:index+1,None]).float().to(device),
              "current_rgb":images[index][None],"past_rgb":images[index-1][None],"current_right_rgb":right[index][None]}
        pre=state is None or reset.pre_reset(age=age,accumulated_update=accumulated)
        if pre: state={"disparity":torch.from_numpy(raw[index-1:index,None]).float().to(device),"valid":torch.from_numpy(valid[index-1:index,None]).bool().to(device)}; age=0; accumulated=0.
        forward=torch.from_numpy(flows[0][index-1:index]).float().to(device); backward=torch.from_numpy(flows[1][index-1:index]).float().to(device)
        evidence,result=h4_infer(h4,extractor,item,state,forward,backward,include_learned=True)
        output[index]=result.fused_disparity[0,0].cpu().numpy(); support[index]=(evidence.aligned_validity&evidence.warp_support)[0,0].cpu().numpy()
        decision=item["raw_valid"]&evidence.aligned_validity&evidence.warp_support; update=float((result.fused_disparity-item["raw"]).abs()[decision].mean()) if bool(decision.any()) else 0.
        state={"disparity":result.fused_disparity,"valid":item["raw_valid"]}; age=advance_state_age(age,reset=pre); accumulated=update if pre else accumulated+update
    return output,support

def self_check(device):
    value=MultiAnchorEvidence(torch.ones(1,1,2,2,device=device),torch.ones(1,4,2,2,device=device),torch.zeros(1,4,2,2,dtype=torch.bool,device=device),torch.ones(1,4,2,2,dtype=torch.bool,device=device),torch.ones(1,4,2,2,device=device),torch.tensor(ANCHOR_AGES,device=device),torch.zeros(4,device=device))
    output=RawMultiAnchorRefiner().to(device).eval()(value); assert torch.isneginf(output.selection_score).all()

def evaluate(args):
    verify_frozen_core(); guard_no_d7(VALIDATION_SEQUENCES); device=torch.device(args.device); self_check(device); models=registered_models(device)
    manifest=json.loads(H4.read_text()); h4_path=Path(manifest["checkpoint_path"]); expected=manifest["checkpoint_sha256"]
    if sha256(h4_path)!=expected: raise RuntimeError("H4 checkpoint hash mismatch")
    h4state=torch.load(h4_path,map_location="cpu",weights_only=False); h4=CODDStyleFusionHead(h4state["cue_channels"]).to(device); h4.load_state_dict(h4state["model"],strict=True); h4.eval(); h4.requires_grad_(False)
    extractor=FrozenResNet18Layer1().to(device); extractor.eval(); adapter=SEARAFTFlowAdapter(device=device); rows=[]; excluded=[]; sequences=VALIDATION_SEQUENCES[:1] if args.smoke else VALIDATION_SEQUENCES; backbones=SEEN_BACKBONES[:1] if args.smoke else SEEN_BACKBONES
    for sequence in sequences:
        print(f"D2 sequence={sequence}",flush=True)
        info=load_sequence_info(sequence); ids=info.frame_ids[:args.max_frames] if args.max_frames else info.frame_ids; images=[]; right=[]; gt=[]; coverage=[]
        for frame_id in ids:
            images.append(rgb_tensor(read_rgb(info.seq_dir/"left"/f"{frame_id}.png"),device)); right.append(rgb_tensor(read_rgb(info.seq_dir/"right"/f"{frame_id}.png"),device)); native,valid_gt=load_frame_gt(info,frame_id); resized,cov=resize_gt_to_cache_masked(native,valid_gt); gt.append(resized); coverage.append(cov)
        gt=np.stack(gt); coverage=np.stack(coverage); indices=list(range(8,len(ids))); h4flows,_=infer_age_flows(adapter,images,list(range(1,len(ids))),[1],args.flow_batch_size,device); ageflows,_=infer_age_flows(adapter,images,indices,list(ANCHOR_AGES),args.flow_batch_size,device)
        for backbone in backbones:
            raw_m,valid_m,cache_ids,_=load_sequence_cache(backbone,sequence)
            if [str(x) for x in cache_ids[:len(ids)]]!=ids: raise RuntimeError(f"frame-ID mismatch: {backbone}/{sequence}")
            raw=np.asarray(raw_m[:len(ids)],dtype=np.float32); valid=np.asarray(valid_m[:len(ids)]).astype(bool); aligned=_align(raw,valid,images,indices,ageflows,device,args.flow_batch_size); h4map,h4support=h4_outputs(raw,valid,gt,coverage,images,right,h4flows[1],h4,extractor,device)
            for offset,index in enumerate(indices):
                evidence=MultiAnchorEvidence(torch.from_numpy(raw[index:index+1,None]).float().to(device),torch.from_numpy(aligned["candidates"][offset:offset+1]).float().to(device),torch.from_numpy(aligned["candidate_valid"][offset:offset+1]).bool().to(device),torch.from_numpy(aligned["warp_support"][offset:offset+1]).bool().to(device),torch.from_numpy(aligned["fb_confidence"][offset:offset+1]).float().to(device),torch.tensor(ANCHOR_AGES,device=device),torch.zeros(4,device=device))
                strict=(coverage[index]>.5)&valid[index]&h4support[index]&(aligned["candidate_valid"][offset]&aligned["warp_support"][offset]).all(axis=0)
                if not strict.any():
                    excluded.append({"backbone":backbone,"sequence":sequence,"frame_id":ids[index],"frame_index":index,"reason":"empty_strict_common_support","base_pixel_count":int(((coverage[index]>.5)&valid[index]&h4support[index]).sum())})
                    continue
                canonical=models[("canonical",0)]; out=canonical["model"](evidence); prediction,_,_,_=retrieve_and_fuse(evidence.raw,evidence,out,probability_threshold=canonical["policy"][0],utility_threshold_px=canonical["policy"][1],hard=False)
                base={"backbone":backbone,"sequence":sequence,"frame_id":ids[index],"frame_index":index,"valid_pixel_count":int(strict.sum()),"raw_error_sum":float(np.abs(raw[index]-gt[index])[strict].sum()),"h4_error_sum":float(np.abs(h4map[index]-gt[index])[strict].sum()),"canonical_error_sum":float(np.abs(prediction[0,0].cpu().numpy()-gt[index])[strict].sum()),"parity_pass":True}
                for (budget,seed), item in models.items():
                    if budget=="canonical": continue
                    out=item["model"](evidence); prediction,_,_,_=retrieve_and_fuse(evidence.raw,evidence,out,probability_threshold=item["policy"][0],utility_threshold_px=item["policy"][1],hard=False); error=float(np.abs(prediction[0,0].cpu().numpy()-gt[index])[strict].sum())
                    rows.append(base|{"budget":budget,"seed":seed,"multi_error_sum":error,"checkpoint_sha256":item["sha256"]})
    if any(row["canonical_error_sum"] is None for row in rows): raise RuntimeError("canonical geometry was not evaluated")
    if args.smoke: print(json.dumps({"status":"PASS","rows":len(rows),"strict_support_exclusions":len(excluded),"dataset_7_opened":False})); return
    write_csv(OUT,rows); atomic_json(CAMPAIGN/"selection/d2_common_support_provenance.json",{"project":"ARGOS v2","status":"PASS","rows":len(rows),"sequences":list(VALIDATION_SEQUENCES),"backbones":list(SEEN_BACKBONES),"strict_common_support":"GT coverage & raw validity & H4 support & all CS1/2/4/8 support; all nine candidate and canonical predictions evaluated","h4_sha256":expected,"canonical_sha256":sha256(CANONICAL),"dataset_7_opened":False})
    write_csv(CAMPAIGN/"selection/d2_common_support_exclusions.csv",excluded)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda:0"); p.add_argument("--flow-batch-size",type=int,default=32); p.add_argument("--max-frames",type=int); p.add_argument("--smoke",action="store_true"); args=p.parse_args(); evaluate(args)
if __name__=="__main__": main()
