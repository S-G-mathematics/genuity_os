import numpy as np
from .factory import TVAEFactory


class TVAEAPI:
    """High-level API for TVAE synthetic tabular data generation (basic edition)"""

    def __init__(self, model_type: str = "basic"):
        # Only basic model is supported in the base package
        if model_type != "basic":
            raise ValueError(
                "This package provides only the basic TVAE. For advanced features, import tvae_premium and use TVAEAPI(model_type='premium')."
            )
        self.model_type = model_type
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
        if all(isinstance(x, int) for x in continuous_cols):
            n_cont = len(continuous_cols)
        else:
            n_cont = len(continuous_cols)
        if all(isinstance(x, int) for x in categorical_cols):
            n_cat = len(categorical_cols)
        else:
            n_cat = len(categorical_cols)
        # Create basic model only
        self.trainer = TVAEFactory.create_basic_model(
            continuous_dims=list(range(n_cont)), categorical_dims=list(range(n_cat))
        )
        losses = self.trainer.fit(
            data, epochs=epochs, verbose=kwargs.get("verbose", True)
        )
        self.is_fitted = True
        return losses

    def generate(self, n_samples: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating samples")
        return self.trainer.generate(n_samples)

    def get_feature_importance(self):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.trainer.get_feature_importance()

    def save(self, filepath: str):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        self.trainer.save_model(filepath)

    def load(self, filepath: str):
        if self.trainer is None:
            raise ValueError("Must create model first (call fit or specify model type)")
        self.trainer.load_model(filepath)
        self.is_fitted = True
