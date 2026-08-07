import importlib
import sys
from pathlib import Path
import torch

from argos_freezed.models.raw_multi_anchor_refiner import MultiAnchorEvidence, RawMultiAnchorRefiner, retrieve_and_fuse

ROOT = Path(__file__).resolve().parents[1]
V2 = Path("/dtu/p1/leopam/ARGOS/ARGOS-V2")


def test_byte_copied_model_matches_original_validated_runner():
    sys.path.insert(0, str(V2))
    original = importlib.import_module("model_design.models.raw_multi_anchor_refiner")
    state = torch.load(ROOT / "checkpoints/raw_multi_anchor_best_validation.pt", map_location="cpu", weights_only=False)
    frozen_model = RawMultiAnchorRefiner(state["channels"], state["blocks"]).eval(); frozen_model.load_state_dict(state["model"])
    original_model = original.RawMultiAnchorRefiner(state["channels"], state["blocks"]).eval(); original_model.load_state_dict(state["model"])
    torch.manual_seed(12); raw = torch.rand(1, 1, 13, 17) + 1; candidates = torch.rand(1, 4, 13, 17) + 1
    valid = torch.rand(1, 4, 13, 17) > .1; support = torch.rand(1, 4, 13, 17) > .05; fb = torch.rand(1, 4, 13, 17)
    args = (raw, candidates, valid, support, fb, torch.tensor([1, 2, 4, 8]), torch.zeros(4))
    frozen_evidence = MultiAnchorEvidence(*args); original_evidence = original.MultiAnchorEvidence(*args)
    frozen_out = frozen_model(frozen_evidence); original_out = original_model(original_evidence)
    assert all(torch.equal(getattr(frozen_out, name), getattr(original_out, name)) for name in frozen_out.__dataclass_fields__)
    frozen_result = retrieve_and_fuse(raw, frozen_evidence, frozen_out, probability_threshold=.9, utility_threshold_px=.1, hard=False)
    original_result = original.retrieve_and_fuse(raw, original_evidence, original_out, probability_threshold=.9, utility_threshold_px=.1, hard=False)
    assert all(torch.equal(a, b) for a, b in zip(frozen_result, original_result))
    chosen = frozen_result[2]
    for frozen_tensor, original_tensor in (
        (torch.gather(frozen_evidence.candidates, 1, chosen), torch.gather(original_evidence.candidates, 1, chosen)),
        (torch.gather(frozen_out.selection_score, 1, chosen), torch.gather(original_out.selection_score, 1, chosen)),
        (torch.gather(frozen_out.fusion_weight, 1, chosen), torch.gather(original_out.fusion_weight, 1, chosen)),
        (torch.gather(frozen_evidence.available, 1, chosen), torch.gather(original_evidence.available, 1, chosen)),
    ):
        assert torch.equal(frozen_tensor, original_tensor)


def test_extracted_alignment_is_tensor_exact():
    sys.path.insert(0, str(V2))
    original = importlib.import_module("model_design.external_components.bidavideo")
    from argos_freezed.alignment.bida_pull_warp import temporal_disparity_evidence
    torch.manual_seed(7); shape = (1, 1, 11, 13)
    current = torch.rand(shape) + 1; past = torch.rand(shape) + 1
    forward = torch.randn(1, 2, 11, 13) * .2; backward = -forward + torch.randn_like(forward) * .01
    valid = torch.rand(shape) > .1; past_valid = torch.rand(shape) > .1
    rgb = torch.rand(1, 3, 11, 13) * 255; past_rgb = torch.rand(1, 3, 11, 13) * 255
    kwargs = dict(current_valid=valid, past_valid=past_valid, current_rgb=rgb, past_rgb=past_rgb)
    frozen = temporal_disparity_evidence(current, past, forward, backward, **kwargs)
    reference = original.temporal_disparity_evidence(current, past, forward, backward, **kwargs)
    assert all(torch.equal(getattr(frozen, name), getattr(reference, name)) for name in frozen.__dataclass_fields__)
