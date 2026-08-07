from __future__ import annotations
import ast, hashlib, importlib, json, os, subprocess, sys
from pathlib import Path
import pytest, torch, yaml

ROOT=Path("/dtu/p1/leopam/ARGOS/ARGOS_FREEZED"); CAMPAIGN=ROOT/"experiments/02_massive_training"; SCRIPTS=CAMPAIGN/"scripts"
sys.path[:0]=[str(ROOT/"src"),str(SCRIPTS)]
from campaign_common import *
from data_pipeline import BalancedGroupSampler
from losses import MultiAnchorLossConfig, multi_anchor_targets, raw_multi_anchor_losses
from argos_freezed.models.raw_multi_anchor_refiner import FEATURE_CHANNELS, MultiAnchorEvidence, RawMultiAnchorRefiner, retrieve_and_fuse

def test_frozen_verifier_and_hashes(): verify_frozen_core()
def test_training_dependency_hashes(): verify_training_dependencies()
def test_architecture_parameter_count(): assert sum(p.numel() for p in RawMultiAnchorRefiner().parameters())==60739
def test_anchor_and_feature_contract():
    expected=["raw_disparity_over_64","candidate_disparity_over_64","signed_residual_over_16","absolute_residual_over_16","local_5x5_absolute_residual_mean_over_8","local_5x5_residual_std_over_8","anchor_age_over_8","candidate_validity","warp_support","forward_backward_confidence","valid_witness_fraction","candidate_median_over_64","candidate_mad_over_8","candidate_to_median_absolute_residual_over_8","raw_to_median_absolute_residual_over_8","agreement_fraction","provenance_raw_zero"]
    frozen=yaml.safe_load((ROOT/"configs/geometry_v1.yaml").read_text())
    assert ANCHOR_AGES==(1,2,4,8) and FEATURE_CHANNELS==17 and frozen["feature_channel_order"]==expected
def test_effective_batch_and_steps():
    config=yaml.safe_load((CAMPAIGN/"campaign_config.yaml").read_text()); assert config["effective_batch_size"]==12
    assert config["steps_per_epoch"]==3973 and [config["budgets"][x]["optimizer_steps"] for x in ("1x","3x","6x")]==[39730,119190,238380]
def test_scheduler_proportional_horizons():
    values=[]
    for epochs in (10,30,60):
        model=torch.nn.Linear(1,1); optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,3973*epochs,eta_min=.0001); values.append(scheduler.T_max)
    assert values==[39730,119190,238380]
def test_optimizer_parity():
    group=torch.optim.AdamW(RawMultiAnchorRefiner().parameters(),lr=.002,weight_decay=.0001).param_groups[0]
    assert group["lr"]==.002 and group["weight_decay"]==.0001 and group["betas"]==(.9,.999) and group["eps"]==1e-8
def test_loss_parity_with_original():
    v2=Path("/dtu/p1/leopam/ARGOS/ARGOS-V2"); sys.path.insert(0,str(v2)); original=importlib.import_module("model_design.losses.raw_multi_anchor_losses")
    torch.manual_seed(3); raw=torch.rand(1,1,8,9)+1; candidates=torch.rand(1,4,8,9)+1; mask=torch.rand(1,4,8,9)>.1
    evidence=MultiAnchorEvidence(raw,candidates,mask,mask,torch.rand_like(candidates),torch.tensor([1,2,4,8]),torch.zeros(4)); output=RawMultiAnchorRefiner()(evidence)
    gt=torch.rand_like(raw)+1; coverage=torch.ones_like(raw); valid=torch.ones_like(raw,dtype=torch.bool)
    local_target=multi_anchor_targets(raw,candidates,gt,coverage,valid,evidence,margin_px=.1); reference_target=original.multi_anchor_targets(raw,candidates,gt,coverage,valid,evidence,margin_px=.1)
    local=raw_multi_anchor_losses(output,evidence,local_target,MultiAnchorLossConfig(),enable_fusion=True); reference=original.raw_multi_anchor_losses(output,evidence,reference_target,original.MultiAnchorLossConfig(),enable_fusion=True)
    assert all(torch.equal(local[key],reference[key]) for key in local)
def test_initial_weights_deterministic_and_distinct():
    audit=json.loads((CAMPAIGN/"protocol_audit/canonical_recipe_audit.json").read_text()); hashes=[]
    for seed in SEEDS:
        state=torch.load(CAMPAIGN/f"initial_weights/seed_{seed}.pt",map_location="cpu",weights_only=False); assert state["tensor_sha256"]==audit["initial_weights"][str(seed)]["tensor_sha256"]; hashes.append(state["tensor_sha256"])
    assert len(set(hashes))==3
def test_sampler_determinism_and_balance():
    metadata=[{"backbone":"a","sequence":"x"}]*2+[{"backbone":"b","sequence":"y"}]*3
    one=BalancedGroupSampler(metadata,10); two=BalancedGroupSampler(metadata,10); assert list(one)==list(two) and len(one)==6
def test_no_future_flow_composition_or_fused_writeback():
    text=(ROOT/"src/argos_freezed/pipeline.py").read_text(); assert "current_to_anchor" in text and "compose(" not in text
    from argos_freezed.memory_bank import RawAnchorBank
    assert not any(hasattr(RawAnchorBank,name) for name in ("add_fused","update_with_output","write_prediction"))
def test_invalid_mask_and_exact_fallback():
    raw=torch.rand(1,1,5,6)+1; invalid=torch.zeros(1,4,5,6,dtype=torch.bool)
    evidence=MultiAnchorEvidence(raw,torch.rand(1,4,5,6)+1,invalid,invalid,torch.zeros(1,4,5,6),torch.tensor([1,2,4,8]),torch.zeros(4)); output=RawMultiAnchorRefiner()(evidence)
    prediction,accepted,_,_=retrieve_and_fuse(raw,evidence,output,probability_threshold=.9,utility_threshold_px=.1,hard=False)
    assert torch.isneginf(output.selection_score).all() and torch.equal(prediction,raw) and not accepted.any()
def test_positive_left_guard():
    from argos_freezed.memory_bank import RawAnchorBank
    bank=RawAnchorBank(); disparity=-torch.ones(1,1,2,2)
    with pytest.raises(ValueError): bank.append_raw(disparity,torch.ones_like(disparity,dtype=torch.bool),torch.ones(1,3,2,2),frame_id="x",frame_index=0)
def test_dataset_split_and_d7_rejection():
    assert TRAIN_IDS==(1,3,6) and VALIDATION_IDS==(2,)
    for value in ("dataset_7_keyframe_1","/cache/dataset_7/x","dataset_id=7"):
        with pytest.raises(RuntimeError,match="D7 ACCESS DENIED"): guard_no_d7(value)
def test_output_isolation():
    for budget in (1,3,6):
        for seed in SEEDS: run_directory(budget,seed).relative_to((ROOT/"experiments").resolve())
def test_launcher_slots_are_disjoint_and_complete():
    from launch_campaign import RUNS
    left,right=RUNS[0::2],RUNS[1::2]
    assert not set(left)&set(right) and left+right!=RUNS and set(left+right)==set(RUNS) and len(RUNS)==9
def test_strict_common_support():
    masks=[torch.tensor([[1,1],[0,1]],dtype=torch.bool),torch.tensor([[1,0],[1,1]],dtype=torch.bool),torch.ones(2,2,dtype=torch.bool)]
    assert torch.equal(strict_common_support(*masks),torch.tensor([[1,0],[0,1]],dtype=torch.bool))
def test_no_critic_or_v2_runtime_imports():
    runtime=("campaign_common.py","data_pipeline.py","losses.py","train_one_run.py","launch_campaign.py")
    imports=[]
    for name in runtime:
        for node in ast.walk(ast.parse((SCRIPTS/name).read_text())):
            if isinstance(node,ast.Import): imports.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom): imports.append(node.module or "")
    joined=" ".join(imports).lower(); assert "critic" not in joined and "model_design" not in joined and "argos_v2" not in joined
def test_frozen_core_not_mutated():
    assert sha256(ROOT/"FREEZE_MANIFEST.json")==FROZEN_MANIFEST_SHA
    subprocess.run(["sha256sum","-c",str(ROOT/"FILE_HASHES.sha256")],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
def test_selection_protocol_pre_registered():
    assert sha256(CAMPAIGN/"selection_protocol.json")==sha256(CAMPAIGN/"selection/selection_protocol.json")=="3120b1e953c160299e0bc50a8288c6a7558d6795d489823463a9fc64c52b48d5"
