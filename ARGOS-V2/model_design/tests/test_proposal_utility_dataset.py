from __future__ import annotations

import pytest
import torch

from model_design.data.proposal_utility_dataset import (
    ProposalUtilityDataset,
    proposal_utility_targets,
    stratified_training_targets,
)


def synthetic() -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.tensor([[[[1.0, 2.0, 4.0, 8.0, 2.0]]]])
    gt = torch.tensor([[[[1.0, 2.0, 3.0, 7.0, 2.0]]]])
    proposal = torch.tensor([[[[1.2, 2.0, 3.4, 8.4, 2.0]]]])
    batch = {
        "raw": raw,
        "gt": gt,
        "gt_coverage": torch.tensor([[[[1.0, 1.0, 1.0, 1.0, 0.2]]]]),
        "raw_valid": torch.ones_like(raw, dtype=torch.bool),
    }
    valid = torch.ones_like(raw, dtype=torch.bool)
    return batch, proposal, valid, valid


def test_utility_and_three_way_labels_match_manual_errors() -> None:
    batch, proposal, aligned, support = synthetic()
    target = proposal_utility_targets(
        batch, proposal, aligned_valid=aligned, warp_support=support,
        epsilon_px=0.1, coverage_threshold=0.5,
    )
    # Raw errors [0,0,1,1], proposal errors [.2,0,.4,1.4].
    torch.testing.assert_close(target.utility[..., :4], torch.tensor([[[[-.2, 0., .6, -.4]]]]))
    assert target.classes.tolist() == [[[[0, 1, 2, 0, -100]]]]
    assert target.helpful.tolist() == [[[[False, False, True, False, False]]]]
    assert target.harmful.tolist() == [[[[True, False, False, True, False]]]]
    assert target.indifferent.tolist() == [[[[False, True, False, False, False]]]]


@pytest.mark.parametrize("epsilon,classes", [
    (0.05, [0, 1, 2, 0, -100]),
    (0.25, [1, 1, 2, 0, -100]),
    (0.50, [1, 1, 2, 1, -100]),
])
def test_epsilon_indifference_ladder(epsilon: float, classes: list[int]) -> None:
    batch, proposal, aligned, support = synthetic()
    target = proposal_utility_targets(
        batch, proposal, aligned_valid=aligned, warp_support=support,
        epsilon_px=epsilon,
    )
    assert target.classes.flatten().tolist() == classes


def test_invalid_candidates_never_enter_targets() -> None:
    batch, proposal, aligned, support = synthetic()
    aligned[..., 0] = False
    support[..., 2] = False
    target = proposal_utility_targets(
        batch, proposal, aligned_valid=aligned, warp_support=support,
        epsilon_px=0.1,
    )
    assert not target.regression_valid[..., 0].any()
    assert not target.regression_valid[..., 2].any()
    assert target.classes[..., 0].item() == -100
    assert target.classes[..., 2].item() == -100


def test_paired_mask_requires_coverage_raw_alignment_and_support() -> None:
    batch, proposal, aligned, support = synthetic()
    batch["raw_valid"][..., 0] = False
    aligned[..., 1] = False
    support[..., 2] = False
    target = proposal_utility_targets(
        batch, proposal, aligned_valid=aligned, warp_support=support,
        epsilon_px=0.1, coverage_threshold=0.5,
    )
    assert target.regression_valid.flatten().tolist() == [False, False, False, True, False]


def test_stratified_sampling_is_deterministic_and_keeps_natural_target_immutable() -> None:
    batch, proposal, aligned, support = synthetic()
    target = proposal_utility_targets(
        batch, proposal, aligned_valid=aligned, warp_support=support,
        epsilon_px=0.1,
    )
    first = stratified_training_targets(target, proposal - batch["raw"], batch["gt"], maximum_pixels=3)
    second = stratified_training_targets(target, proposal - batch["raw"], batch["gt"], maximum_pixels=3)
    assert torch.equal(first.regression_valid, second.regression_valid)
    assert torch.equal(target.regression_valid, torch.tensor([[[[True, True, True, True, False]]]]))


@pytest.mark.parametrize("backbone", ["Fast-FoundationStereo", "CREStereo", "invented"])
def test_unseen_or_unknown_backbones_are_rejected_before_loading(backbone: str) -> None:
    with pytest.raises(ValueError):
        ProposalUtilityDataset([backbone], ["dataset_7_keyframe_1"], max_pairs_per_sequence=1)


def test_real_records_are_exact_causal_and_do_not_cross_or_switch() -> None:
    dataset = ProposalUtilityDataset(
        ["S2M2-S", "RAFT-Stereo"], ["dataset_7_keyframe_1"],
        max_pairs_per_sequence=2, random_clip_start=False,
    )
    for record in dataset.records:
        assert record.current_index == record.past_index + 1
        assert record.sequence == "dataset_7_keyframe_1"
        assert record.backbone in ("S2M2-S", "RAFT-Stereo")
