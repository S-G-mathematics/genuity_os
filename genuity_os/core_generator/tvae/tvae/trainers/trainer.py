import torch
import torch.nn as nn
import torch.optim as optim
from ..models.encoder import TVAEEncoder
from ..models.decoder import TVAEDecoder
from ..samplers.sampler import TVAESampler


class TVAETrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_cont = len(config.continuous_dims)
        self.n_cat = len(config.categorical_dims)
        self.input_dim = self.n_cont + self.n_cat
        self.latent_dim = config.latent_dim
        self.encoder = TVAEEncoder(self.input_dim, self.latent_dim, config).to(
            self.device
        )
        self.decoder = TVAEDecoder(
            self.latent_dim, self.input_dim, config, self.n_cont, self.n_cat
        ).to(self.device)
        self.sampler = TVAESampler(config)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config.learning_rate,
            weight_decay=getattr(config, "weight_decay", 0.0),
        )
        # Basic edition: no premium features
        self.wasserstein_loss = None
        self.kl_annealing = None
        self.gradient_noise = None
        self.quality_gating = None

    def fit(self, data, epochs=1000, verbose=True):
        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        losses = {"recon": [], "kl": [], "total": [], "kl_weight": []}

        for epoch in range(epochs):
            self.encoder.train()
            self.decoder.train()
            self.optimizer.zero_grad()

            mu, logvar = self.encoder(data)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            # Decode (basic)
            recon = self.decoder(z)

            # Reconstruction loss: MSE for continuous, BCEWithLogits for categorical
            if self.n_cat > 0 and self.n_cont > 0:
                recon_cont = recon[:, : self.n_cont]
                recon_cat = recon[:, self.n_cont :]
                target_cont = data[:, : self.n_cont]
                target_cat = data[:, self.n_cont :]
                mse_loss = nn.MSELoss()(recon_cont, target_cont)
                bce_loss = nn.BCEWithLogitsLoss()(recon_cat, target_cat)
                recon_loss = mse_loss + bce_loss
            elif self.n_cat > 0:
                recon_loss = nn.BCEWithLogitsLoss()(recon, data)
            else:
                recon_loss = nn.MSELoss()(recon, data)

            # KL loss (standard Gaussian prior)
            kl_loss = (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
            )

            # KL annealing warmup (basic linear)
            kl_weight = self._compute_kl_weight(epoch, epochs)

            # Apply β-VAE weighting (basic uses beta as-is if set)
            kl_loss = (
                self.config.beta * kl_loss
                if getattr(self.config, "beta", 1.0)
                else kl_loss
            )

            # Monitor and prevent posterior collapse
            if epoch > 50 and kl_loss.item() < 1e-4:
                # If KL is too small, increase the weight temporarily
                kl_weight = min(kl_weight * 2.0, 2.0)

            total_loss = recon_loss + kl_weight * kl_loss

            # No premium annealing or gradient noise in basic

            total_loss.backward()
            self.optimizer.step()

            losses["recon"].append(recon_loss.item())
            losses["kl"].append(kl_loss.item())
            losses["total"].append(total_loss.item())
            losses["kl_weight"].append(kl_weight)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch}: Recon {recon_loss.item():.4f}, KL {kl_loss.item():.4f}, "
                    f"KL Weight {kl_weight:.4f}, Total {total_loss.item():.4f}"
                )
        return losses

    def generate(self, n_samples):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            z = self.sampler.sample(n_samples, device=self.device)
            samples = self.decoder(z)
            samples = samples.cpu().numpy()
            # Robust categorical assignment
            if self.n_cat > 0:
                samples[:, self.n_cont :] = self._assign_categorical(
                    samples[:, self.n_cont :]
                )
            # No quality gating in basic
            return samples

    def save_model(self, filepath):
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
            },
            filepath,
        )

    def load_model(self, filepath):
        import os

        # Security validation
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Check file size (prevent extremely large files)
        file_size = os.path.getsize(filepath)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError(f"Model file too large: {file_size} bytes (max 100MB)")

        # Load with security restrictions
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)

        # Validate checkpoint structure
        required_keys = ["encoder", "decoder"]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Invalid checkpoint: missing {key}")

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])

    def get_feature_importance(self):
        # Placeholder: could use mutual information or decoder weights
        return {}

    def _kl_divergence(self, mu, logvar, prior_mu=None, prior_logvar=None):
        # Standard Gaussian KL for basic edition
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    def _compute_kl_weight(self, epoch, total_epochs):
        """Compute KL weight with warmup to prevent posterior collapse"""
        if epoch < self.config.warmup_epochs:
            # Linear warmup from 0 to max_kl_weight
            return (epoch / self.config.warmup_epochs) * self.config.max_kl_weight
        else:
            return self.config.max_kl_weight

    def _assign_categorical(self, cat_logits):
        # Assign each row to the most probable class for each categorical feature
        return (cat_logits == cat_logits.max(axis=1, keepdims=True)).astype(float)
