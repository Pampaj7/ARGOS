"""Suppression-Only Gate (SOG): a tiny post-hoc correction suppressor for v3.2c.

Design rationale (fable light experimental branch):

The v3.2c refiner has the best accuracy of the line (oracle gap 7.03%, full-GT test
4.6145) but overcorrects on two pathological failure modes (patho new-Bad3 15.77%).
Every attempt to fix that so far traded something structural away:

  * v3.3 threshold-only: safety up, but gap collapses (4.80%) — binary thresholding on
    p_bad cannot separate "good aggressive correction" from "bad aggressive correction".
  * v3.3b fine-tuning v3.2c's residual head: gradients too dilute, nothing moves.
  * v4_tiny retraining from scratch with a damping head: patho new-Bad3 0.33% (the
    mechanism works!) but full-GT test regresses (4.78 > raw 4.67) — a fresh model
    re-derives everything and loses v3.2c's calibrated generalization.

SOG keeps v3.2c *frozen and untouched* and trains only a ~90k-parameter gate that maps
(input features, v3.2c's own p_bad and residual) -> per-pixel suppression s in [0, 1]:

    final = raw + s * (hard_gate_v32c * residual_v32c)

Structural guarantees, by construction rather than by training luck:
  * s ~= 1 at init (sigmoid with +4 logit bias, zero-init head) -> exact v3.2c behavior.
  * s can only *shrink* corrections, never add or grow them -> the final prediction lies
    per-pixel on the segment between raw S2M2 (s=0) and v3.2c (s=1). The v4_tiny failure
    mode (regressing below raw on full-GT) is impossible beyond v3.2c's own errors.
  * Full-GT accuracy loss from suppression is visible to the loss directly (suppressing
    a helpful correction increases GT error), so accuracy/safety is traded per pixel,
    not via a global threshold.

Difference from EGBM: no memory, no experts, no dynamic threshold, no retraining of
detection or correction — one tiny head over a frozen, already-trusted model.
Targets exactly the diagnosed failure (over-aggressive correction inside open gates on
high_temporal_flicker / high_boundary_error content, modified%<->new-Bad3 corr 0.78).

Budget: gate ~90k params (frozen v3.2c base is 195k), combined runtime ~= v3.2c's
1.08 ms/frame + a sub-ms gate pass — far under the 8 ms budget.
"""

from __future__ import annotations

import torch
from torch import nn


def _norm(ch: int) -> nn.Module:
    return nn.GroupNorm(min(8, ch), ch)


class SuppressionGate(nn.Module):
    """Tiny convnet: (features + frozen-refiner outputs) -> suppression map s in [0,1].

    Input channels: 16 (v3 feature stack) + p_bad + residual/scale + |residual|/scale
    + hard gate mask = 20. Head is zero-initialized and the sigmoid gets a +init_bias
    logit, so s ~= sigmoid(init_bias) ~= 0.982 at init: near-exact v3.2c pass-through.
    """

    def __init__(self, in_channels: int = 20, hidden: int = 48, depth: int = 4, init_bias: float = 4.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden, 3, padding=1), _norm(hidden), nn.SiLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), _norm(hidden), nn.SiLU(inplace=True)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("init_bias", torch.tensor(float(init_bias)))

    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.trunk(gate_input)) + self.init_bias)


class V32CWithSuppression(nn.Module):
    """Frozen v3.2c base + trainable suppression gate, exposing the v3 eval interface.

    forward(x, scale) -> (bad_logit, p_bad, suppressed_residual). External eval code
    applies its own hard threshold on p_bad exactly as for v3.2c, so
    refined = raw + (p_bad >= t) * suppressed_residual matches deployment.
    """

    def __init__(self, base: nn.Module, gate: SuppressionGate, threshold: float = 0.7):
        super().__init__()
        self.base = base
        self.gate = gate
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        for p in self.base.parameters():
            p.requires_grad = False

    def gate_input(self, x: torch.Tensor, p_bad: torch.Tensor, residual: torch.Tensor, scale: float) -> torch.Tensor:
        hard = (p_bad >= self.threshold).float()
        return torch.cat([x, p_bad, residual / scale, residual.abs() / scale, hard], dim=1)

    def forward_with_s(self, x: torch.Tensor, scale: float):
        """Returns (bad_logit, p_bad, base_residual, s) — caller applies s as needed."""
        with torch.no_grad():
            bad_logit, p_bad, residual = self.base(x, scale)
        s = self.gate(self.gate_input(x, p_bad, residual, scale))
        return bad_logit, p_bad, residual, s

    def forward(self, x: torch.Tensor, scale: float):
        bad_logit, p_bad, residual, s = self.forward_with_s(x, scale)
        return bad_logit, p_bad, s * residual
