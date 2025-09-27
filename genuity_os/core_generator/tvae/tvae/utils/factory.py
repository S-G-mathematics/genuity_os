from ..config.config import TVAEConfig
from ..trainers.trainer import TVAETrainer


class TVAEFactory:
    """Factory for creating TVAE models"""

    @staticmethod
    def create_basic_model(continuous_dims, categorical_dims):
        config = TVAEConfig(
            continuous_dims=continuous_dims, categorical_dims=categorical_dims
        )
        return TVAETrainer(config)

    # Premium and enterprise builders were moved to tvae_premium
