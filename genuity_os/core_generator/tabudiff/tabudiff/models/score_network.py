"""
Minimal MLP score network for tabular diffusion.
"""

import torch
import torch.nn as nn


class ScoreNetwork(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0
    ):
        super().__init__()
        layers = []
        input_dim = feature_dim + 1  # + time embedding scalar
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, feature_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x_in = torch.cat([x, t], dim=1)
        return self.net(x_in)
