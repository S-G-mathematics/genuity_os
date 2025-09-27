"""
High-level API for TabuDiff basic: train from CSV and generate CSV.
"""

import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ..config.config import TabuDiffConfig
from .factory import TabuDiffFactory


class TabuDiffAPI:
    def __init__(self, config: TabuDiffConfig | None = None):
        self.config = config or TabuDiffConfig()
        self.factory = TabuDiffFactory()
        self.score_model: nn.Module | None = None
        self.scheduler = None
        self.data_mean = None
        self.data_std = None

    def _df_to_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        data = torch.tensor(df.values, dtype=torch.float32, device=self.config.device)

        if self.config.normalize_data:
            # Store normalization parameters for denormalization
            self.data_mean = data.mean(dim=0, keepdim=True)
            self.data_std = (
                data.std(dim=0, keepdim=True) + 1e-8
            )  # Add small epsilon to avoid division by zero
            data = (data - self.data_mean) / self.data_std

        return data

    def fit_dataframe(self, df: pd.DataFrame) -> nn.Module:
        # Input validation
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")

        if df.shape[0] < 10:
            raise ValueError("DataFrame too small (minimum 10 rows)")

        if df.shape[1] < 1:
            raise ValueError("DataFrame must have at least one column")

        # Check for infinite values
        if df.isin([np.inf, -np.inf]).any().any():
            raise ValueError("DataFrame contains infinite values")

        # Check for all-null columns
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            raise ValueError(f"DataFrame contains all-null columns: {null_columns}")

        torch.manual_seed(self.config.seed)

        data = self._df_to_tensor(df)
        feature_dim = data.shape[1]

        self.scheduler = self.factory.create_scheduler(self.config)
        self.score_model = self.factory.create_score_network(self.config, feature_dim)

        optimizer = optim.Adam(
            self.score_model.parameters(), lr=self.config.learning_rate
        )
        self.score_model.train()

        num_batches = max(
            1, (len(data) + self.config.batch_size - 1) // self.config.batch_size
        )

        print(
            f"Training TabuDiff with {self.config.num_epochs} epochs, {num_batches} batches per epoch"
        )

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            perm = torch.randperm(len(data), device=self.config.device)

            for b in range(num_batches):
                idx = perm[
                    b * self.config.batch_size : (b + 1) * self.config.batch_size
                ]
                x0 = data[idx]

                t_indices = torch.randint(
                    0,
                    self.scheduler.num_steps,
                    (x0.size(0),),
                    device=self.config.device,
                )
                betas = self.scheduler.betas[t_indices]
                alpha_bars = self.scheduler.alphas_cumprod[t_indices]

                eps = torch.randn_like(x0)
                xt = (
                    torch.sqrt(alpha_bars).unsqueeze(1) * x0
                    + torch.sqrt(1 - alpha_bars).unsqueeze(1) * eps
                )

                t_cont = t_indices.float() / (self.scheduler.num_steps - 1)
                eps_theta = self.score_model(xt, t_cont.unsqueeze(1))

                loss = torch.mean((eps_theta - eps) ** 2)
                epoch_loss += loss.item()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping for stability
                if self.config.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        self.score_model.parameters(), self.config.gradient_clip_value
                    )

                optimizer.step()

            avg_loss = epoch_loss / num_batches
            if epoch % 10 == 0 or epoch == self.config.num_epochs - 1:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.6f}")

        return self.score_model

    def fit_csv(self, csv_path: str) -> nn.Module:
        df = pd.read_csv(csv_path)
        return self.fit_dataframe(df)

    def generate_dataframe(self, num_samples: int | None = None) -> pd.DataFrame:
        if self.score_model is None or self.scheduler is None:
            raise RuntimeError("Model not trained. Call fit_dataframe or load first.")
        if num_samples is None:
            num_samples = self.config.num_samples

        if num_samples <= 0:
            raise ValueError("Number of samples must be positive")

        if num_samples > 1000000:  # Reasonable limit
            raise ValueError("Number of samples too large (max 1,000,000)")

        print(f"Generating {num_samples} synthetic samples...")

        sampler = self.factory.create_sampler(
            self.config, self.score_model, self.scheduler
        )
        with torch.no_grad():
            samples = sampler.sample(
                num_samples=num_samples,
                feature_dim=self.score_model.net[-1].out_features,
            )

        # Denormalize if data was normalized during training
        if (
            self.config.normalize_data
            and self.data_mean is not None
            and self.data_std is not None
        ):
            samples = samples * self.data_std + self.data_mean

        return pd.DataFrame(samples.cpu().numpy())

    def save(self, path: str | None = None):
        if self.score_model is None:
            raise RuntimeError("Nothing to save. Train or load first.")
        if path is None:
            path = self.config.model_save_path
        torch.save(self.score_model.state_dict(), path)
        return path

    def load(self, feature_dim: int, path: str | None = None) -> nn.Module:
        import os

        if path is None:
            path = self.config.model_save_path

        # Security validation
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Check file size (prevent extremely large files)
        file_size = os.path.getsize(path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError(f"Model file too large: {file_size} bytes (max 100MB)")

        self.score_model = self.factory.create_score_network(self.config, feature_dim)

        # Load with security restrictions
        state_dict = torch.load(
            path, map_location=self.config.device, weights_only=True
        )
        self.score_model.load_state_dict(state_dict)

        self.score_model.eval()
        self.scheduler = self.factory.create_scheduler(self.config)
        return self.score_model

    def fit_csv_and_generate_csv(
        self, input_csv: str, output_csv: str, num_samples: int | None = None
    ) -> str:
        df = pd.read_csv(input_csv)
        self.fit_dataframe(df)
        df_syn = self.generate_dataframe(num_samples=num_samples)
        df_syn.to_csv(output_csv, index=False)
        return output_csv
