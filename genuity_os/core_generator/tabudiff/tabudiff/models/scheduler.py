"""
Variance/beta schedule utilities for diffusion.
"""

import torch


class VarianceScheduler:
    def __init__(
        self, num_steps: int, beta_start: float, beta_end: float, device: str = "cpu"
    ):
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.betas = torch.linspace(
            beta_start, beta_end, steps=num_steps, device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def get_step(self, t: torch.Tensor):
        return self.betas[t], self.alphas[t], self.alphas_cumprod[t]
