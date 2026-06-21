from __future__ import annotations

from torch import nn

from .config import ExperimentConfig
from .modules import (
    FUSION_HEADS,
    MEMORY_MODULES,
    MOTION_ESTIMATORS,
    SEMANTIC_PRIORS,
    STEREO_SOURCES,
    TEACHERS,
    UNCERTAINTY_ESTIMATORS,
    WARPERS,
)
from .types import FusionOutput, TemporalBatch


class PlaygroundModel(nn.Module):
    """Composable temporal stereo refiner assembled entirely from config."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.stereo_source = STEREO_SOURCES.build(config.stereo_source.name, **config.stereo_source.params)
        self.motion_estimator = MOTION_ESTIMATORS.build(config.motion_estimator.name, **config.motion_estimator.params)
        self.warper = WARPERS.build(config.warper.name, **config.warper.params)
        self.short_term_memory = MEMORY_MODULES.build(config.short_term_memory.name, **config.short_term_memory.params)
        self.long_term_memory = MEMORY_MODULES.build(config.long_term_memory.name, **config.long_term_memory.params)
        self.convgru_memory = MEMORY_MODULES.build(config.convgru_memory.name, **config.convgru_memory.params)
        self.uncertainty = UNCERTAINTY_ESTIMATORS.build(config.uncertainty.name, **config.uncertainty.params)
        self.teacher = TEACHERS.build(config.teacher.name, **config.teacher.params)
        self.semantic_prior = SEMANTIC_PRIORS.build(config.semantic_prior.name, **config.semantic_prior.params)
        self.fusion = FUSION_HEADS.build(config.fusion.name, **config.fusion.params)

    def forward(self, batch: TemporalBatch) -> FusionOutput:
        raw = self.stereo_source(batch)
        motion = self.motion_estimator(batch)
        uncertainty = self.uncertainty(batch, raw)
        semantic = self.semantic_prior(batch)
        return self.fusion(batch, raw, motion, self.warper, uncertainty, semantic)
