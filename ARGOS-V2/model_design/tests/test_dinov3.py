from __future__ import annotations

import pytest
import torch

from model_design.external_components import dinov3
from model_design.models.dinov3_memory_selector import (
    DINORepresentationSelector,
    selector_targets,
)


@pytest.fixture(scope="module")
def frozen_dino() -> dinov3.FrozenDINOv3:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return dinov3.FrozenDINOv3(device=device, verify_hash=True)


def test_official_checkpoint_and_architecture_load(frozen_dino) -> None:
    model = frozen_dino.model
    assert dinov3.checkpoint_sha256() == dinov3.EXPECTED_CHECKPOINT_SHA256
    assert model.patch_size == 16
    assert model.embed_dim == 1024
    assert model.n_blocks == 24
    assert model.n_storage_tokens == 4


def test_every_parameter_is_frozen_and_eval(frozen_dino) -> None:
    assert not frozen_dino.training and not frozen_dino.model.training
    assert all(not parameter.requires_grad for parameter in frozen_dino.parameters())
    frozen_dino.train(True)
    assert not frozen_dino.training and not frozen_dino.model.training


def test_token_to_feature_map_and_default_no_graph(frozen_dino) -> None:
    rgb = torch.randint(0, 256, (2, 3, 32, 48), dtype=torch.uint8)
    output = frozen_dino.extract(rgb, layers=(5, 23), input_size=(32, 48))
    assert output.layers == (5, 23)
    assert output.metadata.patch_grid == (2, 3)
    assert [tuple(item.shape) for item in output.feature_maps] == [
        (2, 1024, 2, 3),
        (2, 1024, 2, 3),
    ]
    assert all(not item.requires_grad and item.grad_fn is None for item in output.feature_maps)


def test_intermediate_layer_extraction_is_deterministic(frozen_dino) -> None:
    torch.manual_seed(17)
    rgb = torch.rand(1, 3, 32, 48)
    first = frozen_dino.extract(rgb, layers=(5, 11, 23), input_size=(32, 48))
    second = frozen_dino.extract(rgb, layers=(5, 11, 23), input_size=(32, 48))
    for one, two in zip(first.feature_maps, second.feature_maps, strict=True):
        torch.testing.assert_close(one, two, rtol=0, atol=0)


def test_preprocess_preserves_aspect_ratio_and_normalizes() -> None:
    rgb = torch.zeros(1, 3, 1024, 1280, dtype=torch.uint8)
    image, metadata = dinov3.preprocess_rgb(rgb, (256, 320))
    assert metadata.resized_size == (256, 320)
    assert metadata.padding_ltrb == (0, 0, 0, 0)
    assert metadata.patch_grid == (16, 20)
    expected = -torch.tensor(dinov3.IMAGENET_MEAN) / torch.tensor(dinov3.IMAGENET_STD)
    torch.testing.assert_close(image[0, :, 0, 0], expected)


def test_feature_grid_flow_resize_scales_xy() -> None:
    feature = torch.ones(1, 4, 6, 8)
    flow = torch.ones(1, 2, 3, 4)
    result = dinov3.warp_dino_feature(feature, flow)
    # x and y displacement both become two feature cells.
    expected_support = torch.zeros(1, 1, 6, 8, dtype=torch.bool)
    expected_support[:, :, :4, :6] = True
    assert torch.equal(result.support, expected_support)


def test_bida_feature_warp_identity() -> None:
    feature = torch.randn(2, 7, 4, 6)
    result = dinov3.warp_dino_feature(feature, torch.zeros(2, 2, 4, 6))
    torch.testing.assert_close(result.warped, feature, rtol=1e-6, atol=1e-6)
    assert result.support.all()


def test_known_integer_feature_translation_and_support() -> None:
    feature = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    flow = torch.zeros(1, 2, 3, 4)
    flow[:, 0] = 1
    result = dinov3.warp_dino_feature(feature, flow)
    expected = torch.tensor([[[[1, 2, 3, 0], [5, 6, 7, 0], [9, 10, 11, 0]]]], dtype=torch.float32)
    torch.testing.assert_close(result.warped, expected, rtol=0, atol=0)
    assert result.support[..., :3].all()
    assert not result.support[..., 3].any()


def test_feature_warp_reuses_canonical_bida(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"resize": 0, "warp": 0}
    original_resize = dinov3.resize_flow
    original_warp = dinov3.causal_warp

    def resize(*args, **kwargs):
        calls["resize"] += 1
        return original_resize(*args, **kwargs)

    def warp(*args, **kwargs):
        calls["warp"] += 1
        return original_warp(*args, **kwargs)

    monkeypatch.setattr(dinov3, "resize_flow", resize)
    monkeypatch.setattr(dinov3, "causal_warp", warp)
    dinov3.warp_dino_feature(torch.ones(1, 2, 3, 4), torch.zeros(1, 2, 3, 4))
    assert calls == {"resize": 1, "warp": 1}


def selector_inputs(batch: int = 2):
    geom = torch.randn(batch, 4, 12, 4, 5)
    rgb = torch.rand(batch, 5, 3, 4, 5)
    features = torch.randn(batch, 4, 4, 64, 4, 5)
    valid = torch.ones(batch, 4, 1, 4, 5, dtype=torch.bool)
    return geom, rgb, features, valid


def test_raw_null_option_and_selector_normalization() -> None:
    model = DINORepresentationSelector("P4")
    output = model(*selector_inputs())
    assert output.logits.shape == (2, 5, 4, 5)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(2, 4, 5), atol=1e-6)
    # Zero raw logit is explicit and the initial negative memory bias abstains.
    assert torch.equal(output.logits[:, 0], torch.zeros(2, 4, 5))
    assert (output.probabilities.argmax(dim=1) == 0).all()


def test_selector_contract_has_no_backbone_identity() -> None:
    names = DINORepresentationSelector.forward.__annotations__
    assert "backbone" not in names and "backbone_id" not in names


def test_known_best_memory_target_and_invalid_exclusion() -> None:
    errors = torch.tensor([1.0, 3.0, 0.2, 0.7, 2.0]).view(1, 5, 1, 1, 1).expand(1, 5, 1, 2, 3)
    valid = torch.ones(1, 4, 1, 2, 3, dtype=torch.bool)
    target, useful = selector_targets(errors, valid, margin=0.05)
    assert (target == 2).all()
    assert useful[:, 1].all()
    valid[:, 1] = False
    target, useful = selector_targets(errors, valid, margin=0.05)
    assert (target == 3).all()
    assert not useful[:, 1].any()


def test_selector_gradients_do_not_enter_frozen_descriptors() -> None:
    model = DINORepresentationSelector("P6")
    geom, rgb, features, valid = selector_inputs(1)
    features.requires_grad_(False)
    output = model(geom, rgb, features, valid)
    output.logits.sum().backward()
    assert features.grad is None
    assert any(parameter.grad is not None for parameter in model.ranker.parameters())
