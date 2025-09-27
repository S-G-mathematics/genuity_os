from typing import List, Optional
from ..config.config import CTGANConfig
from ..trainers.trainer import CTGANTrainer


class CTGANFactory:
    """Factory for creating CTGAN models"""

    @staticmethod
    def create_model(
        continuous_dims: List[int],
        categorical_dims: List[int],
        config: Optional[CTGANConfig] = None,
        **kwargs
    ) -> CTGANTrainer:
        """Create CTGAN model with specified configuration"""
        if config is not None:
            # Use provided config object
            return CTGANTrainer(config)
        else:
            # Create new config from parameters
            config = CTGANConfig(
                continuous_dims=continuous_dims,
                categorical_dims=categorical_dims,
                **kwargs
            )
            return CTGANTrainer(config)
