"""
Simple DDPM-like sampler for tabular data.
"""

import torch
import torch.nn.functional as F


class DiffusionSampler:
    def __init__(self, score_model, scheduler, device: str = "cpu"):
        self.score_model = score_model
        self.scheduler = scheduler
        self.device = torch.device(device)

    def p_sample(self, x_t: torch.Tensor, t_index: int) -> torch.Tensor:
        t = torch.full(
            (x_t.size(0),), t_index / (self.scheduler.num_steps - 1), device=self.device
        )
        beta_t = self.scheduler.betas[t_index]
        alpha_t = self.scheduler.alphas[t_index]
        alpha_bar_t = self.scheduler.alphas_cumprod[t_index]

        # Predict noise with score model
        eps_theta = self.score_model(x_t, t.unsqueeze(1))

        # Compute mean of the posterior p(x_{t-1} | x_t)
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
        )

        if t_index > 0:
            z = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_prev = mean + sigma_t * z
        else:
            x_prev = mean
        return x_prev

    def sample(self, num_samples: int, feature_dim: int) -> torch.Tensor:
        x_t = torch.randn(num_samples, feature_dim, device=self.device)
        for t_idx in reversed(range(self.scheduler.num_steps)):
            x_t = self.p_sample(x_t, t_idx)
        return x_t
