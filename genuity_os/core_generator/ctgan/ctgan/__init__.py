"""
CTGAN - Basic synthetic tabular data generation tool

This package provides a clean, optimized implementation of CTGAN (Conditional Tabular GAN)
with improved accuracy and stability for generating high-quality synthetic tabular data.
"""

from .config import CTGANConfig
from .models import CTGANGenerator, CTGANDiscriminator
from .samplers import CTGANSampler
from .trainers import CTGANTrainer
from .utils import CTGANFactory, CTGANAPI

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "CTGANConfig",
    # Models
    "CTGANGenerator",
    "CTGANDiscriminator",
    # Samplers
    "CTGANSampler",
    # Trainers
    "CTGANTrainer",
    # Utilities
    "CTGANFactory",
    "CTGANAPI",
]
