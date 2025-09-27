import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config.config import CTGANConfig


class CTGANDiscriminator(nn.Module):
    """Improved CTGAN discriminator with better accuracy and stability"""

    def __init__(self, config: CTGANConfig):
        super().__init__()
        self.config = config
        input_dim = len(config.continuous_dims) + len(config.categorical_dims)
        d_hidden = config.discriminator_hidden_dim
        d_dropout = config.discriminator_dropout

        # Main discriminator network
        layers = []

        # Input layer
        if config.use_spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Linear(input_dim, d_hidden)))
        else:
            layers.append(nn.Linear(input_dim, d_hidden))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(d_dropout))

        # Hidden layers
        for _ in range(2):
            if config.use_spectral_norm:
                layers.append(
                    nn.utils.spectral_norm(nn.Linear(d_hidden, d_hidden // 2))
                )
            else:
                layers.append(nn.Linear(d_hidden, d_hidden // 2))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(d_dropout))
            d_hidden = d_hidden // 2

        # Output layer
        if config.use_spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Linear(d_hidden, 1)))
        else:
            layers.append(nn.Linear(d_hidden, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator"""
        return torch.sigmoid(self.network(x)).squeeze()

    def compute_gradient_penalty(
        self, real_data: torch.Tensor, fake_data: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP training"""
        if not self.config.use_gradient_penalty:
            return torch.tensor(0.0, device=real_data.device)

        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        # Compute discriminator output for interpolated data
        d_interpolated = self.network(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty
