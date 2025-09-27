"""
Factory for constructing TabuDiff basic components.
"""

import torch
from ..config.config import TabuDiffConfig
from ..models.scheduler import VarianceScheduler
from ..models.score_network import ScoreNetwork
from ..models.sampler import DiffusionSampler


class TabuDiffFactory:
    def create_scheduler(self, config: TabuDiffConfig) -> VarianceScheduler:
        return VarianceScheduler(
            num_steps=config.num_diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            device=config.device,
        )

    def create_score_network(
        self, config: TabuDiffConfig, feature_dim: int
    ) -> ScoreNetwork:
        model = ScoreNetwork(
            feature_dim=feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_hidden_layers,
            dropout=config.dropout,
        )
        return model.to(config.device)

    def create_sampler(
        self, config: TabuDiffConfig, score_model, scheduler
    ) -> DiffusionSampler:
        return DiffusionSampler(
            score_model=score_model, scheduler=scheduler, device=config.device
        )
