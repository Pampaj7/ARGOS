"""Zero-shot OOD model registry for ARGOS temporal refiners.

Each entry pins the model's ALREADY-SELECTED primary-dataset checkpoint, residual
scale, and application policy (no OOD-specific tuning). A model exposes a single
uniform call:

    refined, residual, p_bad, aux = entry.refine(x, raw, device)

where `x` is the 16-channel feature tensor (1,16,H,W) and `raw` is the low-res raw
disparity map (H,W). `refined = raw + applied_residual`.

Two application modes (both are the model's own inference policy, unchanged):
  * "internal_gate": residual is already gated/damped/trusted inside forward()
    (EGBM v1/v2/v2-CARE/v3-CARE-S window, MPC, CPV) -> refined = raw + residual.
  * "threshold_gate": residual applied where p_bad >= stored threshold
    (v3.2c AbstentionCropRefiner) -> refined = raw + (p_bad>=thr)*residual.

Add Agent A's final safe-fraction model by appending one REGISTRY entry; the harness
needs no change.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path("/dtu/p1/leopam/ARGOS")
TRAIN = ROOT / "scripts/temporal_refinement"
for p in (TRAIN, TRAIN / "models", TRAIN / "eval_scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CKPT_ROOT = ROOT / "results/03_temporal_refinement/training"


@dataclass
class ModelEntry:
    name: str
    checkpoint: Path
    build: Callable[[int, float], torch.nn.Module]
    residual_scale: float
    mode: str  # "internal_gate" | "threshold_gate"
    threshold: float | None = None
    temporal: str = "window"  # "window" | "streaming_capable"
    notes: str = ""
    _model: torch.nn.Module | None = field(default=None, repr=False)

    def load(self, device: torch.device) -> torch.nn.Module:
        ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        in_ch = ck.get("input_channels", 16)
        model = self.build(in_ch, self.residual_scale)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[{self.name}] load_state_dict non-strict: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")
        self._model = model.to(device).eval()
        # threshold from checkpoint if present (its own selected policy)
        if self.mode == "threshold_gate" and self.threshold is None:
            self.threshold = float(ck.get("threshold", 0.5))
        return self._model

    @torch.no_grad()
    def refine(self, x: torch.Tensor, raw: torch.Tensor, device: torch.device):
        model = self._model
        out = model(x, self.residual_scale)
        aux: dict[str, Any] = {}
        if self.mode == "internal_gate":
            bad_logit, p_bad, residual = out[0], out[1], out[2]
            applied = residual
            if len(out) > 3 and isinstance(out[3], dict):
                aux = {k: v for k, v in out[3].items() if torch.is_tensor(v)}
        elif self.mode == "threshold_gate":
            bad_logit, p_bad, residual = out[0], out[1], out[2]
            gate = (p_bad >= float(self.threshold)).float()
            applied = residual * gate
            aux = {"threshold_gate": gate}
        else:
            raise ValueError(self.mode)
        refined = raw + applied[0, 0]
        return refined, applied[0, 0], p_bad[0, 0], aux


def _mpc(in_ch: int, scale: float):
    from magnitude_proposal_critic_refiner import magnitude_proposal_critic_refiner
    return magnitude_proposal_critic_refiner(in_ch, scale)


def _cpv(in_ch: int, scale: float):
    from counterfactual_proposal_verifier_refiner import counterfactual_proposal_verifier_refiner
    return counterfactual_proposal_verifier_refiner(in_ch, scale)


def _egbm_v1(in_ch: int, scale: float):
    from experimental_refiner_vx import egbm_refiner
    return egbm_refiner(in_ch, scale)


def _egbm_v2(in_ch: int, scale: float):
    import importlib
    mod = importlib.import_module("egbm_v2_refiner")
    return mod.egbm_refiner(in_ch, scale)


def _egbm_v2_care(in_ch: int, scale: float):
    from egbm_v2_care_refiner import egbm_v2_care
    return egbm_v2_care(in_ch, scale)


def _egbm_v3_care_s(in_ch: int, scale: float):
    from egbm_v3_care_streaming_refiner import egbm_v3_care_streaming
    return egbm_v3_care_streaming(in_ch, scale)


def _v3_2c(in_ch: int, scale: float):
    from train_tiny_refiner_v3_1_staged_abstention import AbstentionCropRefiner
    return AbstentionCropRefiner(in_ch)


def build_registry() -> list[ModelEntry]:
    """Currently-available refiners with their selected primary checkpoints."""
    reg = [
        ModelEntry("v3.2c", CKPT_ROOT / "tiny_refiner_v3_2c_hybrid_oracle_freeze_detector_long/checkpoints/best.pt",
                   _v3_2c, 3.0, "threshold_gate", threshold=0.7,
                   notes="AbstentionCropRefiner; residual applied where p_bad>=0.7 (stored)"),
        ModelEntry("EGBM-v1", CKPT_ROOT / "experimental_refiner_vx_training/checkpoints/best.pt",
                   _egbm_v1, 3.0, "internal_gate"),
        ModelEntry("EGBM-v2", CKPT_ROOT / "egbm_v2_experimental/checkpoints/best.pt",
                   _egbm_v2, 3.0, "internal_gate"),
        ModelEntry("EGBM-v2-CARE", CKPT_ROOT / "egbm_v2_care/checkpoints/best.pt",
                   _egbm_v2_care, 3.0, "internal_gate"),
        ModelEntry("EGBM-v3-CARE-S", CKPT_ROOT / "egbm_v3_care_streaming/checkpoints/best.pt",
                   _egbm_v3_care_s, 3.0, "internal_gate", temporal="streaming_capable",
                   notes="window (causal-replay) mode; streaming degenerate on sparse OOD"),
        ModelEntry("MPC", CKPT_ROOT / "magnitude_proposal_critic_refiner/checkpoints/best_pareto.pt",
                   _mpc, 32.0, "internal_gate", notes="large-proposal; safety-critical OOD"),
        ModelEntry("CPV", CKPT_ROOT / "counterfactual_proposal_verifier_refiner/checkpoints/best_pareto.pt",
                   _cpv, 32.0, "internal_gate", notes="proposal+verifier"),
        # Agent A's final safe-fraction model: append here when available.
    ]
    return [e for e in reg if e.checkpoint.exists()]
