from dataclasses import dataclass
from typing import List


@dataclass
class TVAEConfig:
    """Configuration for the TVAE synthetic tabular data generator"""

    # Basic settings
    latent_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 128
    weight_decay: float = 1e-5

    # VAE-specific settings
    beta: float = 1.0  # β-VAE weight for KL loss
    kl_weight: float = 1.0  # Direct KL loss weight
    warmup_epochs: int = 50  # Epochs for KL annealing warmup
    max_kl_weight: float = 1.0  # Maximum KL weight after warmup

    # Advanced TVAE feature flags
    use_vampprior: bool = False
    use_beta_divergence: bool = False
    use_disentangled_beta_vae: bool = False
    use_mutual_information: bool = False
    use_multi_head_decoder: bool = False
    use_transformer_attention: bool = False
    use_gmm_clustering: bool = False
    use_cyclical_kl: bool = False
    use_gradient_noise: bool = False
    use_wasserstein_loss: bool = False
    use_quality_gating: bool = False

    # Feature dimensions
    continuous_dims: List[int] = None
    categorical_dims: List[int] = None

    def __post_init__(self):
        if self.continuous_dims is None:
            self.continuous_dims = []
        if self.categorical_dims is None:
            self.categorical_dims = []
