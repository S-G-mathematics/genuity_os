"""
Core Generator Module for Genuity

This module contains various synthetic data generation methods including:
- TabuDiff: Advanced tabular diffusion models
- CTGAN: Conditional Tabular GAN
- TVAE: Tabular Variational Autoencoder
- Copula: Copula-based generation
- Masked Predictor: Mask-based generation
- TopoGAN: Topology-aware GAN
- Differential Privacy: Privacy-preserving data processing
"""

from .differential_privacy import (
    DifferentialPrivacyProcessor,
    apply_differential_privacy
)

__all__ = [
    "DifferentialPrivacyProcessor",
    "apply_differential_privacy"
]
