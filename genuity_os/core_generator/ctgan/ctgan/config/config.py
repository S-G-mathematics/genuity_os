from dataclasses import dataclass
from typing import List


@dataclass
class CTGANConfig:
    """Configuration for the basic CTGAN model with improved accuracy settings"""

    # Core architecture settings
    latent_dim: int = 256
    hidden_dim: int = 512
    discriminator_hidden_dim: int = 512  # Match generator hidden dim for better balance
    num_layers: int = 4

    # Training settings - optimized for stability
    learning_rate: float = (
        2e-4  # Slightly faster LR; still stable with spectral norm/GP
    )
    batch_size: int = 256  # Larger batch size for better gradient estimates
    discriminator_dropout: float = 0.2  # Lower dropout to retain capacity

    # Feature dimensions
    continuous_dims: List[int] = None
    categorical_dims: List[int] = None

    # Training stability improvements
    use_spectral_norm: bool = True  # Enable spectral normalization for stability
    use_gradient_penalty: bool = True  # Enable gradient penalty
    gradient_penalty_weight: float = 10.0

    # Loss balancing
    generator_loss_weight: float = 1.0
    discriminator_loss_weight: float = 1.0
    diversity_loss_weight: float = 0.02

    def __post_init__(self):
        if self.continuous_dims is None:
            self.continuous_dims = []
        if self.categorical_dims is None:
            self.categorical_dims = []
