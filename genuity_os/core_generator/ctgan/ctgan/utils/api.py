import numpy as np
from typing import List, Dict
from .factory import CTGANFactory


class CTGANAPI:
    """High-level API for CTGAN synthetic tabular data generation"""

    def __init__(self):
        self.trainer = None
        self.is_fitted = False

    def fit(
        self,
        data: np.ndarray,
        continuous_cols: list,
        categorical_cols: list,
        epochs: int = 1000,
        **kwargs,
    ) -> dict:
        """Fit the CTGAN model to data"""

        # Input validation
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be None or empty")

        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")

        if data.ndim != 2:
            raise ValueError("Data must be 2-dimensional")

        if len(continuous_cols) + len(categorical_cols) != data.shape[1]:
            raise ValueError("Column specification doesn't match data dimensions")

        if epochs <= 0:
            raise ValueError("Epochs must be positive")

        # Check for infinite values
        if np.isinf(data).any():
            raise ValueError("Data contains infinite values")

        # Check for all-null columns
        if np.isnan(data).all(axis=0).any():
            raise ValueError("Data contains all-null columns")

        # Convert column indices to counts
        n_cont = len(continuous_cols)
        n_cat = len(categorical_cols)

        # Extract verbose parameter
        verbose = kwargs.pop("verbose", True)

        # Create model
        self.trainer = CTGANFactory.create_model(
            continuous_dims=list(range(n_cont)),
            categorical_dims=list(range(n_cat)),
            **kwargs,
        )

        # Fit the model
        losses = self.trainer.fit(data, epochs=epochs, verbose=verbose)
        self.is_fitted = True

        return losses

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic samples"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating samples")

        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")

        if n_samples > 1000000:  # Reasonable limit
            raise ValueError("Number of samples too large (max 1,000,000)")

        return self.trainer.generate(n_samples)

    def save(self, filepath: str):
        """Save the model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        self.trainer.save_model(filepath)

    def load(self, filepath: str):
        """Load a saved model"""
        if self.trainer is None:
            # Create a basic trainer if none exists
            # We need to infer the dimensions from the saved model
            import torch

            checkpoint = torch.load(filepath, map_location="cpu")
            config = checkpoint.get("config")
            if config is None:
                raise ValueError("Cannot load model: config not found in saved file")

            self.trainer = CTGANFactory.create_model(
                continuous_dims=config.continuous_dims,
                categorical_dims=config.categorical_dims,
            )

        self.trainer.load_model(filepath)
        self.is_fitted = True
