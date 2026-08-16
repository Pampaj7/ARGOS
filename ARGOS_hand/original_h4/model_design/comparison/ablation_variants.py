"""Head and cue variants for the pre-registered architecture ablations.

`codd_style_fusion.py` is hash-pinned provenance and is not touched. Each variant here
either wraps the pinned cue builder or subclasses the pinned head, changing exactly one
design decision so the comparison against the canonical model stays interpretable.

See `model_design/ablation_preregister.json` for what each variant is supposed to test
and for the yardstick (the three-seed spread) below which differences are not read.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from model_design.models.codd_style_fusion import (
    CODDCues, CODDFusionOutput, CODDStyleFusionHead, _up,
    build_codd_cues as _canonical_build_codd_cues,
)

# Layout of the 142-channel stack, from build_codd_cues. The quarter-resolution block is
# the first 129 channels and its last 64 are the tanh-compressed ResNet-18 features, so
# the appearance block occupies exactly [65:129].
TOTAL_CHANNELS = 142
LOW_CHANNELS = 129
APPEARANCE_CHANNELS = 64
APPEARANCE_SLICE = slice(LOW_CHANNELS - APPEARANCE_CHANNELS, LOW_CHANNELS)   # [65:129]


def build_cues_without_appearance(extractor, **kwargs: Any) -> CODDCues:
    """A1: the canonical cues minus the raw ResNet appearance block (142 -> 78).

    The frozen extractor still runs: appearance self-correlation and cross-appearance
    correlation are computed from the same features and are deliberately retained. This
    ablation asks whether the raw features add anything beyond their own correlations,
    not whether the extractor can be removed --- that is A2.
    """
    if kwargs.get("include_learned_stereo_evidence") is False:
        raise ValueError("A1 is defined on top of the learned-evidence cues; use A2 to remove them")
    cues = _canonical_build_codd_cues(extractor, **kwargs)
    if cues.channels != TOTAL_CHANNELS:
        raise RuntimeError(f"expected {TOTAL_CHANNELS} canonical channels, got {cues.channels}; "
                           "the cue layout changed and APPEARANCE_SLICE is no longer valid")
    values = torch.cat((cues.values[:, :APPEARANCE_SLICE.start],
                        cues.values[:, APPEARANCE_SLICE.stop:]), dim=1)
    expected = TOTAL_CHANNELS - APPEARANCE_CHANNELS
    if values.shape[1] != expected:
        raise RuntimeError(f"A1 produced {values.shape[1]} channels, expected {expected}")
    return CODDCues(values=values, support=cues.support, channels=values.shape[1])


def build_cues_without_learned_evidence(extractor, **kwargs: Any) -> CODDCues:
    """A2: disparity, motion and RGB geometry only (142 -> 38).

    The inherited `step()` always asks for learned stereo evidence, so A2 cannot simply
    pass a null extractor: the pinned builder raises. This forces the flag instead, and
    passes `None` for the extractor so the frozen ResNet is genuinely not evaluated --
    which is the point of A2, and the only variant with a runtime consequence.
    """
    kwargs["include_learned_stereo_evidence"] = False
    cues = _canonical_build_codd_cues(None, **kwargs)
    expected = 38
    if cues.channels != expected:
        raise RuntimeError(f"A2 produced {cues.channels} channels, expected {expected}")
    return cues


class RelaxedConvexityHead(CODDStyleFusionHead):
    """A4: the canonical head plus a zero-initialised additive escape from the interval.

    The canonical output is confined to the segment between the raw disparity and the
    aligned memory. This variant adds one 1x1 convolution on the coarse branch whose
    output is added to that blend, so the head *can* leave the segment. It is initialised
    to zero, so training starts exactly at the canonical model and any departure from
    convexity has to be learned.

    Removing the sigmoid from `fusion_weight` would have been the more obvious ablation
    and would have been wrong: the CODD loss supervises that weight as a probability
    (pushing it to 0 or 1 against the error difference, with a tie regulariser around
    0.5), so an unbounded weight makes those terms degenerate. That would ablate the loss
    rather than the architecture. Here the loss is untouched --- `fusion_weight` and
    `reset_weight` keep their meaning and their supervision, and only the disparity term
    sees the extra degree of freedom.
    """

    def __init__(self, cue_channels: int, width: int = 48) -> None:
        super().__init__(cue_channels, width)
        self.extrapolation = torch.nn.Conv2d(width, 1, 1)
        torch.nn.init.zeros_(self.extrapolation.weight)
        torch.nn.init.zeros_(self.extrapolation.bias)

    def forward(self, cues: CODDCues, raw: torch.Tensor, aligned_memory: torch.Tensor) -> CODDFusionOutput:
        fine = self.full(cues.values)
        coarse = self.coarse(F.avg_pool2d(fine, 2, 2))
        coarse_full = _up(coarse, raw.shape[-2:])
        reset = torch.sigmoid(self.reset(torch.cat((fine, coarse_full), dim=1)))
        fusion_low = torch.sigmoid(self.fusion(coarse))
        fusion = _up(fusion_low, raw.shape[-2:])
        temporal = reset * fusion
        convex = (1.0 - temporal) * raw + temporal * aligned_memory
        escape = _up(self.extrapolation(coarse), raw.shape[-2:])
        return CODDFusionOutput(reset, fusion, temporal, convex + escape)


class SingleResolutionHead(CODDStyleFusionHead):
    """A3: the canonical head with the context branch moved to full resolution.

    The canonical head gathers context in a 48-channel branch running at a quarter of
    the cache grid, and predicts the fusion weight there. This variant runs exactly the
    same branch at full resolution instead: the average pooling is dropped and the
    strided convolution becomes stride one.

    Nothing else changes, and crucially the parameter count is *identical* --- convolution
    weights do not depend on spatial size. Removing the branch outright would have cut the
    head from 177k to about 41k parameters and confounded 'context needs low resolution'
    with 'the head needs capacity'. Here the only thing that moves is the receptive field,
    which shrinks by a factor of four in pixels.
    """

    def __init__(self, cue_channels: int, width: int = 48) -> None:
        super().__init__(cue_channels, width)
        strided = self.coarse[0]
        same = torch.nn.Conv2d(strided.in_channels, strided.out_channels, 3,
                               stride=1, padding=1, bias=False)
        same.weight = strided.weight            # the same parameter tensor, not a copy
        self.coarse[0] = same

    def forward(self, cues: CODDCues, raw: torch.Tensor, aligned_memory: torch.Tensor) -> CODDFusionOutput:
        fine = self.full(cues.values)
        context = self.coarse(fine)                     # full resolution: no pool, no stride
        reset = torch.sigmoid(self.reset(torch.cat((fine, context), dim=1)))
        fusion = torch.sigmoid(self.fusion(context))    # already full resolution
        temporal = reset * fusion
        return CODDFusionOutput(reset, fusion, temporal,
                                (1.0 - temporal) * raw + temporal * aligned_memory)


def demo() -> None:
    """Self-check: A1 removes exactly the appearance block, A4 starts as the canonical head."""
    channels, height, width = TOTAL_CHANNELS, 32, 40
    values = torch.arange(channels, dtype=torch.float32).view(1, channels, 1, 1).expand(1, channels, height, width)
    cues = CODDCues(values=values.contiguous(), support=torch.ones(1, 1, height, width, dtype=torch.bool),
                    channels=channels)

    kept = torch.cat((values[:, :APPEARANCE_SLICE.start], values[:, APPEARANCE_SLICE.stop:]), dim=1)
    assert kept.shape[1] == 78, kept.shape
    present = {int(v) for v in kept[0, :, 0, 0]}
    assert present == set(range(65)) | set(range(129, 142)), "A1 removed the wrong channel range"

    raw = torch.full((1, 1, height, width), 8.0)
    memory = torch.full((1, 1, height, width), 12.0)
    canonical = CODDStyleFusionHead(channels).eval()
    relaxed = RelaxedConvexityHead(channels).eval()
    relaxed.load_state_dict(canonical.state_dict(), strict=False)
    with torch.inference_mode():
        a = canonical(cues, raw, memory).fused_disparity
        b = relaxed(cues, raw, memory).fused_disparity
    assert torch.allclose(a, b, atol=1e-6), "A4 must start identical to the canonical head"
    extra = sum(p.numel() for p in relaxed.parameters()) - sum(p.numel() for p in canonical.parameters())
    assert extra == 49, extra

    single = SingleResolutionHead(channels).eval()
    assert sum(p.numel() for p in single.parameters()) == sum(p.numel() for p in canonical.parameters()), \
        "A3 must keep the canonical parameter count; only the receptive field may change"
    with torch.inference_mode():
        out = single(cues, raw, memory)
    assert out.fusion_weight.shape[-2:] == raw.shape[-2:], "A3 predicts fusion at full resolution"
    print(f"OK: A1 keeps 78 channels and drops [{APPEARANCE_SLICE.start}:{APPEARANCE_SLICE.stop}]; "
          f"A4 adds {extra} parameters and is initially identical to the canonical head")


if __name__ == "__main__":
    demo()
