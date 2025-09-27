import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Union
import logging

from ..config.config import CTGANConfig
from ..models.generator import CTGANGenerator
from ..models.discriminator import CTGANDiscriminator
from ..samplers.sampler import CTGANSampler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CTGANTrainer:
    """Improved CTGAN trainer with better accuracy and stability"""

    def __init__(self, config: CTGANConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.generator = CTGANGenerator(config).to(self.device)
        self.discriminator = CTGANDiscriminator(config).to(self.device)
        self.sampler = CTGANSampler(config)

        # Optimizers with better settings
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999)
        )

        # Loss tracking
        self.losses = defaultdict(list)

    def compute_diversity_loss(self, generated_samples: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss to prevent mode collapse"""
        # Compute pairwise distances
        distances = torch.cdist(generated_samples, generated_samples, p=2)

        # Encourage diversity by maximizing minimum distance
        min_distances = torch.min(
            distances + torch.eye(distances.size(0), device=self.device) * 1e6, dim=1
        )[0]
        diversity_loss = -torch.mean(min_distances)

        return diversity_loss

    def train_step(self, real_data: torch.Tensor, epoch: int) -> Dict[str, float]:
        """Single training step with improved stability"""
        batch_size = real_data.size(0)
        losses = {}

        # Sample latent codes
        z = self.sampler.sample(batch_size, self.device)

        # Generate fake data
        fake_data = self.generator(z)

        # Train Discriminator
        self.d_optimizer.zero_grad()

        # Real data discrimination
        real_scores = self.discriminator(real_data)
        fake_scores = self.discriminator(fake_data.detach())

        # Compute discriminator losses
        d_loss_real = F.binary_cross_entropy(real_scores, torch.ones_like(real_scores))
        d_loss_fake = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))

        d_loss = d_loss_real + d_loss_fake

        # Add gradient penalty if enabled
        if self.config.use_gradient_penalty:
            gradient_penalty = self.discriminator.compute_gradient_penalty(
                real_data, fake_data.detach()
            )
            d_loss += self.config.gradient_penalty_weight * gradient_penalty
            losses["gradient_penalty"] = gradient_penalty.item()

        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator (every other step for stability)
        if epoch % 2 == 0:
            self.g_optimizer.zero_grad()

            # Generate new fake data
            z = self.sampler.sample(batch_size, self.device)
            fake_data = self.generator(z)

            # Get discriminator scores for fake data
            fake_scores = self.discriminator(fake_data)

            # Compute generator loss
            g_loss = F.binary_cross_entropy(fake_scores, torch.ones_like(fake_scores))

            # Add diversity loss
            diversity_loss = self.compute_diversity_loss(fake_data)
            g_loss += self.config.diversity_loss_weight * diversity_loss

            g_loss.backward()
            self.g_optimizer.step()

            losses["generator_loss"] = g_loss.item()
            losses["diversity_loss"] = diversity_loss.item()

        # Store losses
        losses.update(
            {
                "discriminator_loss": d_loss.item(),
                "d_loss_real": d_loss_real.item(),
                "d_loss_fake": d_loss_fake.item(),
            }
        )

        return losses

    def fit(
        self, real_data: np.ndarray, epochs: int = 1000, verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the model with improved stability"""
        real_data = np.array(real_data, dtype=np.float32)
        real_tensor = torch.tensor(real_data, dtype=torch.float32).to(self.device)

        # Training loop
        for epoch in range(epochs):
            # Create batches
            n_samples = real_tensor.size(0)
            indices = torch.randperm(n_samples)

            epoch_losses = defaultdict(list)

            for i in range(0, n_samples, self.config.batch_size):
                end_idx = min(i + self.config.batch_size, n_samples)
                batch_indices = indices[i:end_idx]
                batch_data = real_tensor[batch_indices]

                # Skip if batch too small
                if batch_data.size(0) < 2:
                    continue

                # Training step
                step_losses = self.train_step(batch_data, epoch)

                # Accumulate losses
                for key, value in step_losses.items():
                    epoch_losses[key].append(value)

            # Average epoch losses
            for key, values in epoch_losses.items():
                if values:
                    avg_loss = np.mean(values)
                    self.losses[key].append(avg_loss)

            # Verbose logging
            if verbose and epoch % 100 == 0:
                log_msg = f"Epoch {epoch}:"
                for key, values in self.losses.items():
                    if values:
                        log_msg += f" {key}={values[-1]:.4f}"
                logger.info(log_msg)

        return dict(self.losses)

    def generate(
        self, n_samples: int, return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Generate synthetic samples"""
        self.generator.eval()

        with torch.no_grad():
            # Sample latent codes
            z = self.sampler.sample(n_samples, self.device)

            # Generate samples with hard categorical for inference
            fake_data = self.generator(z, hard_categorical=True)

        self.generator.train()

        if return_numpy:
            return fake_data.cpu().numpy()
        return fake_data

    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save(
            {
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "config": self.config,
                "losses": dict(self.losses),
            },
            filepath,
        )

    def load_model(self, filepath: str):
        """Load a trained model with security validation"""
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
        required_keys = ["generator_state_dict", "discriminator_state_dict"]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Invalid checkpoint: missing {key}")

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.losses = defaultdict(list, checkpoint.get("losses", {}))
