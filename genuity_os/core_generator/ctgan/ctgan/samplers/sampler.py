import torch
import numpy as np
from ..config.config import CTGANConfig


class CTGANSampler:
    """Basic CTGAN sampler with improved stability"""

    def __init__(self, config: CTGANConfig):
        self.config = config
        self.latent_dim = config.latent_dim

    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Sample latent codes from standard normal distribution"""
        return torch.randn(n_samples, self.latent_dim, device=device)
