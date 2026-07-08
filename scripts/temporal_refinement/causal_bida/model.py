"""Forward-only causal BiDAStabilizer adaptations for ARGOS v2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .official_blocks import ResidualBlocksWithInputConv, flow_warp


@dataclass
class CausalBiDAState:
    hidden: torch.Tensor
    prev_raw: torch.Tensor | None = None
    prev_rgb: torch.Tensor | None = None

    def detach(self) -> "CausalBiDAState":
        return CausalBiDAState(
            hidden=self.hidden.detach(),
            prev_raw=None if self.prev_raw is None else self.prev_raw.detach(),
            prev_rgb=None if self.prev_rgb is None else self.prev_rgb.detach(),
        )


class FaithfulCausalBiDA(nn.Module):
    """Official BiDAStabilizer core converted to forward-only streaming.

    The official stabilizer builds a 3-channel local disparity stack
    [prev-aligned, current, next-aligned], extracts 48-channel local features,
    propagates hidden features forward/backward with separate 5-block residual
    encoders, fuses both directions, then predicts an additive residual.

    This causal baseline keeps the official channel counts and blocks, removes
    future/backward propagation, and uses [prev-aligned, current, current].
    """

    def __init__(
        self,
        mid_channels: int = 48,
        num_blocks: int = 5,
        residual_bound: float | None = None,
        identity_init: bool = True,
    ):
        super().__init__()
        self.mid_channels = mid_channels
        self.residual_bound = residual_bound
        self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + mid_channels, mid_channels, num_blocks
        )
        self.fusion = nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=True)
        self.conv_hr = nn.Conv2d(mid_channels, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if identity_init:
            nn.init.zeros_(self.conv_last.weight)
            nn.init.zeros_(self.conv_last.bias)

    def init_state(self, batch_size: int, height: int, width: int, device=None, dtype=None) -> CausalBiDAState:
        hidden = torch.zeros(batch_size, self.mid_channels, height, width, device=device, dtype=dtype)
        return CausalBiDAState(hidden=hidden)

    def detach_state(self, state: CausalBiDAState) -> CausalBiDAState:
        return state.detach()

    def _bounded(self, delta: torch.Tensor) -> torch.Tensor:
        if self.residual_bound is None:
            return delta
        return self.residual_bound * torch.tanh(delta)

    def step(
        self,
        current_rgb: torch.Tensor | None,
        current_raw_disparity: torch.Tensor,
        previous_rgb: torch.Tensor | None,
        previous_raw_disparity: torch.Tensor | None,
        flow_target_to_previous: torch.Tensor | None,
        reliability_mask: torch.Tensor | None,
        state: CausalBiDAState | None,
    ) -> tuple[torch.Tensor, CausalBiDAState, dict[str, torch.Tensor]]:
        raw = current_raw_disparity
        b, _, h, w = raw.shape
        if state is None:
            state = self.init_state(b, h, w, raw.device, raw.dtype)

        if previous_raw_disparity is None or flow_target_to_previous is None:
            prev_warp = raw
            hidden_warp = state.hidden
        else:
            prev_warp = flow_warp(previous_raw_disparity, flow_target_to_previous)
            hidden_warp = flow_warp(state.hidden, flow_target_to_previous)

        local = torch.cat([prev_warp, raw, raw], dim=1)
        feat = self.feat_extract(local)
        hidden = self.forward_resblocks(torch.cat([feat, hidden_warp], dim=1))
        out = self.lrelu(self.fusion(hidden))
        out = self.lrelu(self.conv_hr(out))
        delta = self._bounded(self.conv_last(out))
        refined = raw + delta
        next_state = CausalBiDAState(hidden=hidden, prev_raw=raw, prev_rgb=current_rgb)
        return refined, next_state, {"delta": delta, "hidden": hidden, "prev_warp": prev_warp}

    def forward_sequence(
        self,
        raw: torch.Tensor,
        flow_target_to_previous: torch.Tensor | None = None,
        rgb: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        reset_state: bool = True,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        b, t, _, h, w = raw.shape
        state = self.init_state(b, h, w, raw.device, raw.dtype) if reset_state else None
        outs, diags = [], []
        for i in range(t):
            flow = None if i == 0 or flow_target_to_previous is None else flow_target_to_previous[:, i - 1]
            prev_raw = None if i == 0 else raw[:, i - 1]
            prev_rgb = None if i == 0 or rgb is None else rgb[:, i - 1]
            cur_rgb = None if rgb is None else rgb[:, i]
            rel = None if valid is None else valid[:, i]
            refined, state, diag = self.step(cur_rgb, raw[:, i], prev_rgb, prev_raw, flow, rel, state)
            outs.append(refined)
            diags.append(diag)
        return torch.stack(outs, 1), diags


class SafeCausalBiDA(FaithfulCausalBiDA):
    """ARGOS v2 safety wrapper over the causal BiDA core."""

    def __init__(
        self,
        mid_channels: int = 48,
        num_blocks: int = 5,
        residual_bound: float = 3.0,
        gate_bias: float = -4.0,
        use_rgb: bool = True,
    ):
        super().__init__(mid_channels, num_blocks, residual_bound=residual_bound, identity_init=True)
        self.use_rgb = use_rgb
        gate_in = mid_channels + 2 + (6 if use_rgb else 0)
        self.gate_head = nn.Sequential(
            nn.Conv2d(gate_in, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, gate_bias)

    def step(
        self,
        current_rgb: torch.Tensor | None,
        current_raw_disparity: torch.Tensor,
        previous_rgb: torch.Tensor | None,
        previous_raw_disparity: torch.Tensor | None,
        flow_target_to_previous: torch.Tensor | None,
        reliability_mask: torch.Tensor | None,
        state: CausalBiDAState | None,
    ) -> tuple[torch.Tensor, CausalBiDAState, dict[str, torch.Tensor]]:
        raw = current_raw_disparity
        refined_base, next_state, diag = super().step(
            current_rgb,
            raw,
            previous_rgb,
            previous_raw_disparity,
            flow_target_to_previous,
            reliability_mask,
            state,
        )
        delta = diag["delta"]
        rel = torch.ones_like(raw) if reliability_mask is None else reliability_mask
        gate_parts = [next_state.hidden, rel, (diag["prev_warp"] - raw).abs()]
        if self.use_rgb:
            if current_rgb is None:
                current_rgb = raw.new_zeros(raw.shape[0], 3, raw.shape[-2], raw.shape[-1])
            if previous_rgb is None or flow_target_to_previous is None:
                prev_rgb_warp = current_rgb
            else:
                prev_rgb_warp = flow_warp(previous_rgb, flow_target_to_previous)
            gate_parts.extend([current_rgb, (current_rgb - prev_rgb_warp).abs()])
        gate = torch.sigmoid(self.gate_head(torch.cat(gate_parts, dim=1)))
        safe_delta = gate * delta
        refined = raw + safe_delta
        diag.update({"gate": gate, "delta": safe_delta, "proposal_delta": delta})
        return refined, next_state, diag


class AlignedLocalOnlyFaithful(nn.Module):
    """Causal aligned-local-only baseline.

    This isolates BiDAStabilizer's local aligned disparity evidence without
    persistent propagation state. The official first block is adapted from
    3 disparity channels [prev,current,next] to 2 channels [prev,current].
    """

    def __init__(self, mid_channels: int = 48, residual_bound: float | None = None, identity_init: bool = True):
        super().__init__()
        self.mid_channels = mid_channels
        self.residual_bound = residual_bound
        self.feat_extract = ResidualBlocksWithInputConv(2, mid_channels, 5)
        self.fusion = nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=True)
        self.conv_hr = nn.Conv2d(mid_channels, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if identity_init:
            nn.init.zeros_(self.conv_last.weight)
            nn.init.zeros_(self.conv_last.bias)

    def init_state(self, batch_size: int, height: int, width: int, device=None, dtype=None):
        return None

    def detach_state(self, state):
        return None

    def _bounded(self, delta: torch.Tensor) -> torch.Tensor:
        if self.residual_bound is None:
            return delta
        return self.residual_bound * torch.tanh(delta)

    def step(
        self,
        current_rgb: torch.Tensor | None,
        current_raw_disparity: torch.Tensor,
        previous_rgb: torch.Tensor | None,
        previous_raw_disparity: torch.Tensor | None,
        flow_target_to_previous: torch.Tensor | None,
        reliability_mask: torch.Tensor | None,
        state,
    ) -> tuple[torch.Tensor, None, dict[str, torch.Tensor]]:
        raw = current_raw_disparity
        if previous_raw_disparity is None or flow_target_to_previous is None:
            prev_warp = raw
        else:
            prev_warp = flow_warp(previous_raw_disparity, flow_target_to_previous)
        feat = self.feat_extract(torch.cat([prev_warp, raw], dim=1))
        out = self.lrelu(self.fusion(feat))
        out = self.lrelu(self.conv_hr(out))
        delta = self._bounded(self.conv_last(out))
        return raw + delta, None, {"delta": delta, "prev_warp": prev_warp}

    def forward_sequence(
        self,
        raw: torch.Tensor,
        flow_target_to_previous: torch.Tensor | None = None,
        rgb: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        reset_state: bool = True,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        outs, diags = [], []
        for i in range(raw.shape[1]):
            flow = None if i == 0 or flow_target_to_previous is None else flow_target_to_previous[:, i - 1]
            prev_raw = None if i == 0 else raw[:, i - 1]
            cur_rgb = None if rgb is None else rgb[:, i]
            prev_rgb = None if i == 0 or rgb is None else rgb[:, i - 1]
            rel = None if valid is None else valid[:, i]
            refined, _, diag = self.step(cur_rgb, raw[:, i], prev_rgb, prev_raw, flow, rel, None)
            outs.append(refined)
            diags.append(diag)
        return torch.stack(outs, 1), diags


class AlignedLocalOnlySafe(AlignedLocalOnlyFaithful):
    """Aligned-local-only baseline with ARGOS v2 bounded gated residual."""

    def __init__(
        self,
        mid_channels: int = 48,
        residual_bound: float = 3.0,
        gate_bias: float = -4.0,
        use_rgb: bool = True,
    ):
        super().__init__(mid_channels=mid_channels, residual_bound=residual_bound, identity_init=True)
        self.use_rgb = use_rgb
        gate_in = mid_channels + 2 + (6 if use_rgb else 0)
        self.gate_head = nn.Sequential(
            nn.Conv2d(gate_in, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, gate_bias)

    def step(
        self,
        current_rgb: torch.Tensor | None,
        current_raw_disparity: torch.Tensor,
        previous_rgb: torch.Tensor | None,
        previous_raw_disparity: torch.Tensor | None,
        flow_target_to_previous: torch.Tensor | None,
        reliability_mask: torch.Tensor | None,
        state,
    ) -> tuple[torch.Tensor, None, dict[str, torch.Tensor]]:
        raw = current_raw_disparity
        refined_base, _, diag = super().step(
            current_rgb,
            raw,
            previous_rgb,
            previous_raw_disparity,
            flow_target_to_previous,
            reliability_mask,
            state,
        )
        proposal = diag["delta"]
        rel = torch.ones_like(raw) if reliability_mask is None else reliability_mask
        gate_parts = [diag["prev_warp"] - raw, rel]
        feat = self.feat_extract(torch.cat([diag["prev_warp"], raw], dim=1))
        gate_parts.insert(0, feat)
        if self.use_rgb:
            if current_rgb is None:
                current_rgb = raw.new_zeros(raw.shape[0], 3, raw.shape[-2], raw.shape[-1])
            if previous_rgb is None or flow_target_to_previous is None:
                prev_rgb_warp = current_rgb
            else:
                prev_rgb_warp = flow_warp(previous_rgb, flow_target_to_previous)
            gate_parts.extend([current_rgb, (current_rgb - prev_rgb_warp).abs()])
        gate = torch.sigmoid(self.gate_head(torch.cat(gate_parts, dim=1)))
        delta = gate * proposal
        diag.update({"gate": gate, "proposal_delta": proposal, "delta": delta})
        return raw + delta, None, diag
