"""
Configuration for TabuDiff Basic.
"""

from dataclasses import dataclass


@dataclass
class TabuDiffConfig:
    # Training
    learning_rate: float = 5e-4  # Reduced for better convergence
    batch_size: int = 128  # Smaller batch size for better gradient estimates
    num_epochs: int = 50  # Increased epochs for better training
    seed: int = 42

    # Diffusion
    num_diffusion_steps: int = 1000  # More steps for better quality
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Model
    hidden_dim: int = 512  # Larger model for better capacity
    num_hidden_layers: int = 4  # More layers
    dropout: float = 0.1  # Add dropout for regularization

    # Generation
    num_samples: int = 1000
    device: str = "cpu"

    # IO
    model_save_path: str = "tabudiff_basic_score_network.pt"

    # Data preprocessing
    normalize_data: bool = True  # Enable data normalization
    clip_gradients: bool = True  # Enable gradient clipping
    gradient_clip_value: float = 1.0  # Gradient clipping threshold
