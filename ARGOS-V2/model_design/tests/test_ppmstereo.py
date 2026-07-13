from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from model_design.external_components import bidavideo as bida
from model_design.external_components import ppmstereo as ppm
from model_design.models.learned_t1_refiner import LearnedT1Refiner
from model_design.models.learned_ppm_selector import LearnedPPMSelectorRefiner


def entry(sequence: str, index: int, value: float = 1.0) -> ppm.MemoryEntry:
    disparity = torch.full((1, 1, 3, 4), value)
    return ppm.MemoryEntry(
        sequence_id=sequence,
        frame_index=index,
        age=0,
        disparity=disparity,
        validity=torch.ones_like(disparity, dtype=torch.bool),
        rgb=torch.zeros(1, 3, 3, 4),
    )


def test_memory_reset_and_sequence_boundary() -> None:
    bank = ppm.CausalMemoryBank()
    bank.append(entry("a", 0))
    bank.append(entry("a", 1))
    assert len(bank.entries) == 2
    bank.append(entry("b", 0))
    assert bank.sequence_id == "b"
    assert [item.frame_index for item in bank.entries] == [0]
    bank.reset()
    assert bank.entries == () and bank.sequence_id is None


def test_causal_ordering_and_no_future_access() -> None:
    bank = ppm.CausalMemoryBank()
    bank.append(entry("s", 2))
    with pytest.raises(ValueError, match="strictly causal"):
        bank.append(entry("s", 2))
    bank.append(entry("s", 4))
    assert bank.candidates(4) == (replace_age(bank.entries[0], 2),)
    assert bank.candidates(2) == ()


def replace_age(item: ppm.MemoryEntry, age: int) -> ppm.MemoryEntry:
    from dataclasses import replace

    return replace(item, age=age)


def test_exact_age_handling_and_recent_order() -> None:
    bank = ppm.CausalMemoryBank()
    for index in range(9):
        bank.append(entry("s", index, float(index)))
    candidates = bank.candidates(9, ages=(1, 2, 4, 8))
    assert [item.age for item in candidates] == [1, 2, 4, 8]
    assert [item.frame_index for item in candidates] == [8, 7, 5, 1]


def test_topk_correctness_and_deterministic_age_ties() -> None:
    scores = torch.tensor([[0.4, 0.9, 0.9, 0.1]])
    ages = torch.tensor([1, 8, 2, 4])
    out = ppm.deterministic_topk(scores, 3, ages=ages)
    assert out.indices.tolist() == [[2, 1, 0]]
    out2 = ppm.deterministic_topk(scores, 3, ages=ages)
    assert torch.equal(out.indices, out2.indices)


def test_invalid_memory_exclusion_and_zero_weights() -> None:
    scores = torch.tensor([[100.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    valid = torch.tensor([[False, True, True], [False, False, False]])
    selection = ppm.deterministic_topk(scores, 2, candidate_valid=valid)
    weights = ppm.normalized_play_weights(selection)
    assert selection.indices[0].tolist() == [1, 2]
    assert torch.allclose(weights[0].sum(), torch.tensor(1.0))
    assert torch.equal(weights[1], torch.zeros(2))


def test_play_weight_normalization() -> None:
    selection = ppm.deterministic_topk(torch.tensor([[3.0, 2.0, 1.0]]), 3)
    weights = ppm.normalized_play_weights(selection, temperature=0.5)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))
    assert weights[0, 0] > weights[0, 1] > weights[0, 2]


def test_spatially_aligned_redundancy_uses_intersection() -> None:
    features = torch.tensor(
        [[[[[1.0, 0.0]]], [[[1.0, 99.0]]], [[[0.0, 1.0]]]]]
    )  # [1,3,1,1,2]
    support = torch.tensor(
        [[[[[True, False]]], [[[True, False]]], [[[True, False]]]]]
    )
    matrix, pair_valid = ppm.spatial_redundancy_matrix(features, support)
    assert pair_valid.all()
    assert torch.allclose(matrix[0, 0, 1], torch.tensor(1.0))
    assert torch.allclose(matrix[0, 0, 2], torch.tensor(0.0))


def test_aggregation_is_validity_aware_and_differentiable() -> None:
    memory = torch.tensor([[[[[2.0, 2.0]]], [[[6.0, 6.0]]]]], requires_grad=True)
    validity = torch.tensor([[[[[True, False]]], [[[True, True]]]]])
    selection = ppm.deterministic_topk(torch.tensor([[1.0, 1.0]]), 2)
    weights = torch.tensor([[0.5, 0.5]], requires_grad=True)
    result = ppm.aggregate_selected_memory(memory, validity, selection, weights)
    assert torch.allclose(result.value, torch.tensor([[[[4.0, 6.0]]]]))
    assert result.valid.all()
    result.value.sum().backward()
    assert memory.grad is not None and memory.grad.abs().sum() > 0
    assert weights.grad is not None and weights.grad.abs().sum() > 0


def test_bida_alignment_calls_canonical_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"count": 0}
    source = entry("s", 3)
    shape = source.disparity.shape

    def fake_evidence(*args, **kwargs):
        called["count"] += 1
        one = torch.ones(shape)
        return SimpleNamespace(
            aligned_past_disparity=one * 7,
            warp_support=one.bool(),
            aligned_validity=one.bool(),
            forward_backward_error=one,
            forward_backward_confidence=one,
            photometric_residual=one,
            absolute_disparity_disagreement=one,
            signed_disparity_disagreement=one,
            flow_magnitude=one,
        )

    monkeypatch.setattr(bida, "temporal_disparity_evidence", fake_evidence)
    result = ppm.align_entry_with_bida(
        source,
        current_frame_index=5,
        current_disparity=torch.ones(shape),
        current_validity=torch.ones(shape, dtype=torch.bool),
        current_rgb=torch.zeros(1, 3, 3, 4),
        flow_current_to_memory=torch.zeros(1, 2, 3, 4),
        flow_memory_to_current=torch.zeros(1, 2, 3, 4),
    )
    assert called["count"] == 1
    assert result.age == 2 and torch.all(result.aligned_disparity == 7)


def test_memory_contract_has_no_backbone_identity() -> None:
    names = {field.name for field in fields(ppm.MemoryEntry)}
    assert "backbone" not in names and "backbone_id" not in names


def test_synthetic_known_best_memory_is_selected() -> None:
    raw = torch.zeros(1, 1, 2, 2)
    candidates = torch.stack(
        [torch.full_like(raw, 5.0), torch.full_like(raw, 0.2), torch.full_like(raw, 3.0)], dim=1
    )
    target = torch.full_like(raw, 0.25)
    errors = (candidates - target[:, None]).abs().mean(dim=(2, 3, 4))
    scores = -errors
    selected = ppm.deterministic_topk(scores, 1, ages=torch.tensor([1, 4, 8]))
    assert selected.indices.item() == 1


def test_identity_initialization_reuses_validated_bounded_refiner() -> None:
    model = LearnedT1Refiner(variant="A2")
    batch = 2
    raw = torch.rand(batch, 1, 8, 10) * 10
    evidence = {
        "aligned_past_disparity": torch.rand(batch, 1, 8, 10) * 10,
        "current_valid": torch.ones(batch, 1, 8, 10),
        "aligned_validity": torch.ones(batch, 1, 8, 10),
        "warp_support": torch.ones(batch, 1, 8, 10),
    }
    output = model(raw, evidence)
    assert torch.allclose(output.disparity, raw, atol=1e-7)
    assert output.update.abs().max() <= model.tau_px + 1e-7


def test_faithful_similarity_matches_original_method() -> None:
    torch.manual_seed(4)
    q = torch.randn(1, 8, 3, 8, 12)
    k = torch.randn_like(q)
    original_class = ppm._import_original_ppm_class()
    expected = original_class.compute_qk_similarity(None, q, k, t=3)
    actual = ppm.compute_qk_similarity_faithful(q, k, t=3)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_memory_bank_is_deterministic() -> None:
    def run() -> list[tuple[int, int]]:
        bank = ppm.CausalMemoryBank(max_entries=5)
        for index in range(10):
            bank.append(entry("s", index))
        return [(item.frame_index, item.age) for item in bank.candidates(10, (1, 2, 4, 8))]

    assert run() == run() == [(9, 1), (8, 2), (6, 4)]


def test_learned_ppm_identity_and_gradients() -> None:
    torch.manual_seed(3)
    model = LearnedPPMSelectorRefiner(channels=8)
    raw = torch.rand(2, 1, 6, 8) * 8
    candidates = torch.rand(2, 4, 1, 6, 8) * 8
    ones = torch.ones(2, 4, 1, 6, 8)
    evidence = {
        "aligned_past_disparity": candidates,
        "aligned_validity": ones.bool(),
        "warp_support": ones.bool(),
        "forward_backward_error": ones * 0.1,
        "forward_backward_confidence": ones * 0.9,
        "photometric_residual": ones * 0.1,
        "flow_magnitude": ones,
    }
    output = model(raw, torch.ones_like(raw).bool(), evidence, torch.tensor([1, 2, 4, 8]))
    assert torch.allclose(output.disparity, raw, atol=1e-7)
    assert torch.allclose(
        output.raw_abstain_weight + output.play_weights.sum(dim=1), torch.ones_like(raw), atol=1e-6
    )
    output.candidate_logits.sum().backward()
    assert any(parameter.grad is not None for parameter in model.selector.parameters())


def test_learned_selector_can_rank_known_best_memory() -> None:
    model = LearnedPPMSelectorRefiner(channels=8)
    optimizer = torch.optim.Adam(model.selector.parameters(), lr=0.05)
    raw = torch.zeros(1, 1, 4, 4)
    candidates = torch.stack(
        [torch.full_like(raw, 5.0), torch.full_like(raw, 1.0), torch.full_like(raw, 3.0)], dim=1
    )
    ones = torch.ones_like(candidates)
    evidence = {
        "aligned_past_disparity": candidates,
        "aligned_validity": ones.bool(),
        "warp_support": ones.bool(),
        "forward_backward_error": ones * 0.1,
        "forward_backward_confidence": ones * 0.9,
        "photometric_residual": ones * 0.1,
        "flow_magnitude": ones,
    }
    for _ in range(20):
        optimizer.zero_grad()
        output = model(raw, torch.ones_like(raw).bool(), evidence, torch.tensor([1, 2, 4]))
        loss = torch.nn.functional.cross_entropy(
            output.candidate_logits[:, :, 0], torch.ones(1, 4, 4, dtype=torch.long)
        )
        loss.backward()
        optimizer.step()
    output = model(raw, torch.ones_like(raw).bool(), evidence, torch.tensor([1, 2, 4]))
    assert output.candidate_logits[:, 1].mean() > output.candidate_logits[:, 0].mean()
