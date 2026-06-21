from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModuleConfig:
    """Declarative module selection.

    `name="none"` means the component is intentionally disabled.
    """

    name: str = "none"
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Any, default: str = "none") -> "ModuleConfig":
        if obj is None:
            return cls(default, {})
        if isinstance(obj, str):
            return cls(obj, {})
        if isinstance(obj, dict):
            name = str(obj.get("name", default))
            params = dict(obj.get("params", {}))
            extra = {k: v for k, v in obj.items() if k not in {"name", "params"}}
            params.update(extra)
            return cls(name, params)
        raise TypeError(f"Unsupported module config: {obj!r}")


@dataclass(frozen=True)
class WorkflowConfig:
    mode: str = "smoke"
    sequence_length: int = 5
    batch_size: int = 1
    crop_height: int = 64
    crop_width: int = 96
    amp: bool = False
    bf16: bool = False
    ddp: bool = False
    full_sequence_validation: bool = True
    early_abort: bool = True
    resume: str | None = None


@dataclass(frozen=True)
class LossWeights:
    gt: float = 0.0
    bad2_proxy: float = 0.0
    teacher: float = 0.5
    spatial_teacher: float = 0.0
    raw_fidelity: float = 0.1
    temporal: float = 0.1
    motion_compensated: float = 0.1
    residual: float = 0.05
    alpha_prior: float = 0.0
    reset: float = 0.0
    uncertainty: float = 0.0
    source_weight_entropy: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    stereo_source: ModuleConfig = field(default_factory=lambda: ModuleConfig("s2m2_s512"))
    motion_estimator: ModuleConfig = field(default_factory=lambda: ModuleConfig("zero"))
    warper: ModuleConfig = field(default_factory=lambda: ModuleConfig("identity"))
    short_term_memory: ModuleConfig = field(default_factory=lambda: ModuleConfig("none"))
    long_term_memory: ModuleConfig = field(default_factory=lambda: ModuleConfig("none"))
    convgru_memory: ModuleConfig = field(default_factory=lambda: ModuleConfig("none"))
    uncertainty: ModuleConfig = field(default_factory=lambda: ModuleConfig("none"))
    teacher: ModuleConfig = field(default_factory=lambda: ModuleConfig("stereoanyvideo"))
    semantic_prior: ModuleConfig = field(default_factory=lambda: ModuleConfig("none"))
    fusion: ModuleConfig = field(default_factory=lambda: ModuleConfig("raw_s2m2_s"))
    losses: LossWeights = field(default_factory=LossWeights)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            experiment_name=str(data["experiment_name"]),
            stereo_source=ModuleConfig.from_obj(data.get("stereo_source"), "s2m2_s512"),
            motion_estimator=ModuleConfig.from_obj(data.get("motion_estimator"), "zero"),
            warper=ModuleConfig.from_obj(data.get("warper"), "identity"),
            short_term_memory=ModuleConfig.from_obj(data.get("short_term_memory"), "none"),
            long_term_memory=ModuleConfig.from_obj(data.get("long_term_memory"), "none"),
            convgru_memory=ModuleConfig.from_obj(data.get("convgru_memory"), "none"),
            uncertainty=ModuleConfig.from_obj(data.get("uncertainty"), "none"),
            teacher=ModuleConfig.from_obj(data.get("teacher"), "stereoanyvideo"),
            semantic_prior=ModuleConfig.from_obj(data.get("semantic_prior"), "none"),
            fusion=ModuleConfig.from_obj(data.get("fusion"), "raw_s2m2_s"),
            losses=LossWeights(**dict(data.get("losses", {}))),
            workflow=WorkflowConfig(**dict(data.get("workflow", {}))),
        )

    def to_dict(self) -> dict[str, Any]:
        def module_to_dict(module: ModuleConfig) -> dict[str, Any]:
            return {"name": module.name, "params": dict(module.params)}

        return {
            "experiment_name": self.experiment_name,
            "stereo_source": module_to_dict(self.stereo_source),
            "motion_estimator": module_to_dict(self.motion_estimator),
            "warper": module_to_dict(self.warper),
            "short_term_memory": module_to_dict(self.short_term_memory),
            "long_term_memory": module_to_dict(self.long_term_memory),
            "convgru_memory": module_to_dict(self.convgru_memory),
            "uncertainty": module_to_dict(self.uncertainty),
            "teacher": module_to_dict(self.teacher),
            "semantic_prior": module_to_dict(self.semantic_prior),
            "fusion": module_to_dict(self.fusion),
            "losses": self.losses.__dict__,
            "workflow": self.workflow.__dict__,
        }


def load_experiment_config(path: Path) -> ExperimentConfig:
    with path.open() as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return ExperimentConfig.from_dict(payload)
