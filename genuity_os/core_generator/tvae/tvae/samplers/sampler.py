import torch


class TVAESampler:
    def __init__(self, config):
        self.config = config
        self.latent_dim = config.latent_dim
        # Basic edition: no GMM dependency

    def set_gmm(self, gmm):
        """No-op in basic edition (GMM available in tvae_premium)."""
        return None

    def sample(self, n_samples, device=None, use_gmm=False):
        """Sample latent vectors for generation.

        Args:
            n_samples (int): Number of samples to generate
            device (torch.device): Torch device
            use_gmm (bool): Whether to use GMM if available

        Returns:
            torch.Tensor: Latent vectors [n_samples, latent_dim]
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Basic: standard normal sampling only
        return torch.randn(n_samples, self.latent_dim).to(device)
