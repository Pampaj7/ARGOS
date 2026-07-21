from __future__ import annotations

import hashlib
import inspect
import sys
from collections import Counter
from pathlib import Path

import pytest
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from model_design.data.multidomain_raw_error_dataset import (
    D4DAnchorDataset,
    DomainBalancedSampler,
    FrozenMultiDomainPredictions,
    MultiDomainRawErrorDataset,
    MultiDomainRecord,
    manifest_digest,
    stratified_raw_error_targets,
    verify_geometric_gt_path,
)
from model_design.data.raw_error_dataset import RawErrorTargets
from model_design.losses.raw_error_losses import RawErrorLossConfig, raw_error_losses
from model_design.models.abstention import authorized_update
from model_design.models.learned_t1_refiner import LearnedT1Refiner
from model_design.models.raw_error_detector import RawErrorDetector, RawErrorEvidence
from run_multidomain_raw_error_training import (
    PRIMARY_UNSEEN_BACKBONE,
    SEEN_BACKBONES,
    parse_backbone_list,
    sha256,
    split_manifest,
    verify_frozen_manifest,
)


class TinyDomain(Dataset):
    def __init__(self, domain: str, groups: list[tuple[str, str]], count: int = 3) -> None:
        self.records = []
        for backbone, sequence in groups:
            for frame in range(count):
                self.records.append(MultiDomainRecord(
                    domain=domain, backbone=backbone, sequence=sequence, specimen=sequence,
                    past_frame_id=f"{frame:04d}", current_frame_id=f"{frame + 1:04d}",
                    source_index=len(self.records), supervision_source=f"{domain} GT",
                ))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "domain": record.domain, "backbone": record.backbone,
            "sequence": record.sequence, "specimen": record.specimen,
            "past_frame_id": record.past_frame_id,
            "current_frame_id": record.current_frame_id,
        }


def tiny_combined() -> MultiDomainRawErrorDataset:
    return MultiDomainRawErrorDataset({
        "SCARED-C": TinyDomain("SCARED-C", [("S2M2-S", "s1"), ("RAFT-Stereo", "s2")]),
        "D4D": TinyDomain("D4D", [("S2M2-S", "d1"), ("S2M2-S", "d2")]),
    })


def test_domain_ratio_backbone_and_sequence_balancing_are_exact() -> None:
    dataset = tiny_combined()
    sampler = DomainBalancedSampler(
        dataset, {"SCARED-C": .75, "D4D": .25}, samples_per_epoch=40, seed=7,
    )
    indices = list(sampler)
    records = [dataset.records[index] for index in indices]
    assert Counter(record.domain for record in records) == {"SCARED-C": 30, "D4D": 10}
    scared = Counter((record.backbone, record.sequence) for record in records if record.domain == "SCARED-C")
    d4d = Counter(record.sequence for record in records if record.domain == "D4D")
    assert max(scared.values()) - min(scared.values()) <= 1
    assert max(d4d.values()) - min(d4d.values()) <= 1


def test_sampler_and_manifest_are_deterministic_and_resume_by_epoch() -> None:
    dataset = tiny_combined()
    first = DomainBalancedSampler(dataset, {"SCARED-C": .5, "D4D": .5}, samples_per_epoch=24, seed=19)
    second = DomainBalancedSampler(dataset, {"SCARED-C": .5, "D4D": .5}, samples_per_epoch=24, seed=19)
    assert list(first) == list(second)
    first.set_epoch(3); second.set_epoch(3)
    assert list(first) == list(second)
    assert manifest_digest(dataset.records) == manifest_digest(dataset.records)


def test_each_sample_has_one_domain_backbone_and_exact_causal_order() -> None:
    dataset = tiny_combined()
    for index, record in enumerate(dataset.records):
        sample = dataset[index]
        assert sample["domain"] == record.domain
        assert sample["backbone"] == record.backbone
        assert sample["sequence"] == record.sequence == record.specimen
        assert int(sample["current_frame_id"]) == int(sample["past_frame_id"]) + 1


def test_heterogeneous_samples_have_an_identical_collation_contract() -> None:
    dataset = tiny_combined()
    first = dataset[0]
    second_domain = next(i for i, record in enumerate(dataset.records) if record.domain == "D4D")
    second = dataset[second_domain]
    assert set(first) == set(second)
    assert first["past_index"] == second["past_index"] == -1


def test_real_d4d_split_is_specimen_disjoint_and_anchor_local() -> None:
    train = D4DAnchorDataset(["specimen_1"])
    calibration = D4DAnchorDataset(["specimen_2"])
    test = D4DAnchorDataset(["specimen_3"])
    assert (len(train), len(calibration), len(test)) == (72, 33, 51)
    assert {record.specimen for record in train.records} == {"specimen_1"}
    assert {record.specimen for record in calibration.records} == {"specimen_2"}
    assert {record.specimen for record in test.records} == {"specimen_3"}
    assert set(r.current_frame_id for r in train.records).isdisjoint(r.current_frame_id for r in calibration.records)
    assert all(r.supervision_source == "Zivid structured-light curated anchor" for r in train.records)
    assert all(r.current_frame_id != r.past_frame_id for r in train.records)
    sample = train[0]
    assert all(torch.isfinite(sample[key]).all() for key in ("raw", "past", "gt", "gt_coverage"))
    assert torch.equal(sample["gt_valid"], sample["gt_coverage"] > .5)
    valid = sample["gt_valid"] & sample["raw_valid"]
    assert (sample["raw"] - sample["gt"])[valid].abs().mean() < 10


@pytest.mark.parametrize("name", ["stereo_depth.npy", "IGEV++_prediction.npy", "igev/output.npy"])
def test_prediction_derived_d4d_sources_are_never_accepted_as_gt(name: str) -> None:
    with pytest.raises(ValueError, match="forbidden as GT"):
        verify_geometric_gt_path(Path("dataset/D4D") / name, "D4D")


def test_ood_prediction_cache_is_read_only_and_excludes_unseen_backbones(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="training backbones"):
        FrozenMultiDomainPredictions("CREStereo", "D4D", cache_root=tmp_path)
    directory = tmp_path / "RAFT-Stereo" / "D4D"
    directory.mkdir(parents=True)
    np.save(directory / "disparity.npy", np.ones((2, 144, 180), np.float16))
    np.save(directory / "valid_mask.npy", np.ones((2, 144, 180), np.uint8))
    np.save(directory / "frame_ids.npy", np.array(["a", "b"]))
    (directory / ".complete").write_text("ok")
    cache = FrozenMultiDomainPredictions("RAFT-Stereo", "D4D", cache_root=tmp_path)
    disparity, valid = cache.get("b")
    assert disparity.shape == (144, 180) and valid.all()
    with pytest.raises(KeyError, match="absent"):
        cache.get("not-present")


def test_m1_manifest_predeclares_all_leakage_barriers() -> None:
    manifest = split_manifest("m1")
    assert manifest["added_domain"]["train"] == ["specimen_1"]
    assert manifest["added_domain"]["calibration"] == ["specimen_2"]
    assert manifest["added_domain"]["final_test"] == ["specimen_3"]
    assert manifest["fully_unseen_domain"] == "SERV-CT"
    assert "SERV-CT" in manifest["forbidden_before_freeze"]
    assert PRIMARY_UNSEEN_BACKBONE in manifest["forbidden_before_freeze"]
    assert "CREStereo" in manifest["forbidden_before_freeze"]
    assert set(manifest["scared_c"]["backbones"]) == set(SEEN_BACKBONES)


def test_added_domain_backbone_protocol_is_explicit_and_rejects_invalid_values() -> None:
    assert parse_backbone_list("S2M2-S") == ("S2M2-S",)
    assert parse_backbone_list("S2M2-S,RAFT-Stereo") == ("S2M2-S", "RAFT-Stereo")
    with pytest.raises(Exception):
        parse_backbone_list("Fast-FoundationStereo")
    manifest = split_manifest("m1", added_backbones=("S2M2-S",))
    assert manifest["added_domain"]["backbones"] == ["S2M2-S"]


def test_final_only_load_contract_is_explicit_in_runner() -> None:
    import run_multidomain_raw_error_training as runner

    fitting_source = inspect.getsource(runner.make_sources)
    final_source = inspect.getsource(runner.final_evaluation)
    assert "Fast-FoundationStereo" not in fitting_source
    assert "CREStereo" not in fitting_source
    assert 'if args.fold=="m1"' in final_source
    assert "SERV-CT-unseen" in final_source
    assert final_source.index("verify_frozen_manifest") < final_source.index("SERV-CT-unseen")


def test_stratified_sampling_preserves_targets_and_balances_error_bins() -> None:
    error = torch.tensor([[[[.1, .2, 1., 2., 4., 6.]]]])
    valid = torch.ones_like(error, dtype=torch.bool)
    targets = RawErrorTargets(error, (error > .5).float(), valid, valid, error <= .5)
    sampled = stratified_raw_error_targets(targets, pixels_per_bin=1, seed=4)
    assert torch.equal(sampled.error, error)
    assert sampled.regression_valid.sum() == 3
    selected = error[sampled.regression_valid]
    assert (selected <= .5).sum() == 1
    assert ((selected > .5) & (selected <= 3)).sum() == 1
    assert (selected > 3).sum() == 1


def test_frozen_proposal_and_cache_evidence_receive_no_gradients() -> None:
    a2 = LearnedT1Refiner("A2").eval().requires_grad_(False)
    detector = RawErrorDetector("s1", channels=24)
    raw = torch.rand(1, 1, 3, 4)
    raw.requires_grad_(False)
    one = torch.ones_like(raw)
    evidence = RawErrorEvidence(
        raw=raw, raw_valid=one.bool(), aligned=raw + .1,
        aligned_valid=one.bool(), warp_support=one.bool(),
        forward_backward_error=one * .1, forward_backward_confidence=one * .9,
        photometric_residual=one * .1, flow_magnitude=one,
        a2_update=one * .2, a2_error_gate=one * .3, a2_memory_gate=one * .4,
    )
    output = detector(evidence)
    target_error = torch.rand_like(output.mu)
    valid = torch.ones_like(target_error, dtype=torch.bool)
    targets = RawErrorTargets(target_error, (target_error > .5).float(), valid, valid, target_error <= .5)
    losses = raw_error_losses(output, targets, RawErrorLossConfig(mode="a4", false_positive_cost=5))
    losses["total"].backward()
    assert all(parameter.grad is None for parameter in a2.parameters())
    assert raw.grad is None
    assert torch.isfinite(losses["total"])


def test_abstention_is_bit_exact_and_paired_mask_is_shared() -> None:
    raw = torch.tensor([[[[1., 2., 3.]]]])
    update = torch.tensor([[[[.5, -1., 2.]]]])
    rejected = authorized_update(raw, update, torch.zeros_like(raw, dtype=torch.bool))
    assert torch.equal(rejected, raw)
    accepted = authorized_update(raw, update, torch.ones_like(raw, dtype=torch.bool))
    assert torch.equal(accepted, raw + update)
    common = torch.tensor([[[[True, False, True]]]])
    assert torch.equal(common, common.clone())  # one object is supplied to both paired metrics


def test_frozen_manifest_hash_verification_detects_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "detector.pt"
    artifact.write_bytes(b"frozen")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    expected = hashlib.sha256(b"frozen").hexdigest()
    (frozen / "frozen_manifest.json").write_text(
        '{"artifacts":{"detector":{"path":"%s","sha256":"%s"}}}' % (artifact, expected)
    )
    verify_frozen_manifest(frozen)
    assert sha256(artifact) == expected
    artifact.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_frozen_manifest(frozen)


def test_multidomain_source_contains_no_cache_writes_or_backbone_identity_feature() -> None:
    import model_design.data.multidomain_raw_error_dataset as module

    source = inspect.getsource(module)
    assert "np.save(" not in source and "torch.save(" not in source
    assert "backbone_identity" not in source and "backbone_embedding" not in source
