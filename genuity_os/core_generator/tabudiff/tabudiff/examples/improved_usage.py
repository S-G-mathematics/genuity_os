"""
Improved TabuDiff usage with better configuration and data preprocessing.
This should significantly improve the quality of generated data.
"""

import pandas as pd
import numpy as np
from tabudiff.basic.utils import TabuDiffAPI
from tabudiff.basic.config import TabuDiffConfig


def create_sample_data():
    """Create a sample dataset with realistic structure"""
    np.random.seed(42)
    n_samples = 2000  # More data for better training

    # Generate correlated features
    age = np.random.normal(35, 12, n_samples)
    age = np.clip(age, 18, 80)  # Clip to realistic range

    # Income correlated with age
    income = 15000 + age * 800 + np.random.normal(0, 8000, n_samples)
    income = np.clip(income, 10000, 200000)  # Clip to realistic range

    # Score with some correlation to income
    score = 20 + (income / 1000) * 0.3 + np.random.normal(0, 15, n_samples)
    score = np.clip(score, 0, 100)  # Clip to 0-100 range

    df = pd.DataFrame({"age": age, "income": income, "score": score})

    return df


def improved_training_example():
    """Demonstrate improved TabuDiff training"""

    # Create sample data
    df = create_sample_data()

    print("=" * 60)
    print("IMPROVED TABUDIFF TRAINING EXAMPLE")
    print("=" * 60)

    print("\nOriginal data statistics:")
    print(df.describe())
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Original data correlations:")
    print(df.corr().round(3))

    # Improved configuration for better results
    config = TabuDiffConfig(
        # Training parameters
        learning_rate=2e-4,  # Lower learning rate for stability
        batch_size=128,  # Moderate batch size
        num_epochs=200,  # More epochs for convergence
        seed=42,
        # Diffusion parameters
        num_diffusion_steps=1000,  # More steps for better quality
        beta_start=1e-4,
        beta_end=0.02,
        # Model architecture
        hidden_dim=512,  # Larger model
        num_hidden_layers=4,  # More layers
        dropout=0.1,  # Regularization
        # Data preprocessing
        normalize_data=True,  # Enable normalization
        clip_gradients=True,  # Enable gradient clipping
        gradient_clip_value=1.0,
        # Generation
        num_samples=1000,
        device="cpu",
    )

    print(f"\nConfiguration:")
    print(f"- Learning rate: {config.learning_rate}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Epochs: {config.num_epochs}")
    print(f"- Diffusion steps: {config.num_diffusion_steps}")
    print(f"- Hidden dim: {config.hidden_dim}")
    print(f"- Normalize data: {config.normalize_data}")

    # Train the model
    print(f"\n{'='*20} TRAINING {'='*20}")
    api = TabuDiffAPI(config)
    api.fit_dataframe(df)

    # Generate synthetic data
    print(f"\n{'='*20} GENERATION {'='*20}")
    synthetic_df = api.generate_dataframe(num_samples=1000)

    # Compare results
    print(f"\n{'='*20} RESULTS COMPARISON {'='*20}")

    print("\nSynthetic data statistics:")
    print(synthetic_df.describe())
    print(f"\nSynthetic data shape: {synthetic_df.shape}")
    print(f"Synthetic data correlations:")
    print(synthetic_df.corr().round(3))

    # Calculate quality metrics
    print(f"\n{'='*20} QUALITY METRICS {'='*20}")

    # Mean absolute percentage error for statistics
    orig_stats = df.describe()
    synth_stats = synthetic_df.describe()

    mape_scores = []
    for col in df.columns:
        orig_mean = orig_stats.loc["mean", col]
        synth_mean = synth_stats.loc["mean", col]
        mape = abs(synth_mean - orig_mean) / orig_mean * 100
        mape_scores.append(mape)
        print(f"{col}: MAPE = {mape:.2f}%")

    avg_mape = np.mean(mape_scores)
    print(f"\nAverage MAPE: {avg_mape:.2f}%")

    if avg_mape < 10:
        print("✅ EXCELLENT quality (MAPE < 10%)")
    elif avg_mape < 20:
        print("✅ GOOD quality (MAPE < 20%)")
    elif avg_mape < 30:
        print("⚠️  FAIR quality (MAPE < 30%)")
    else:
        print("❌ POOR quality (MAPE > 30%)")

    return synthetic_df


def quick_fix_for_your_data(df):
    """Quick fix function for your existing data"""

    print("=" * 60)
    print("QUICK FIX FOR YOUR DATA")
    print("=" * 60)

    # Improved configuration
    config = TabuDiffConfig(
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=150,  # More epochs
        num_diffusion_steps=1000,  # More diffusion steps
        hidden_dim=256,
        num_hidden_layers=3,
        dropout=0.1,
        normalize_data=True,  # Enable normalization
        clip_gradients=True,  # Enable gradient clipping
    )

    print(f"Training with improved configuration...")
    print(f"- Epochs: {config.num_epochs}")
    print(f"- Diffusion steps: {config.num_diffusion_steps}")
    print(f"- Normalization: {config.normalize_data}")

    api = TabuDiffAPI(config)
    api.fit_dataframe(df)

    synthetic_df = api.generate_dataframe(num_samples=500)

    print("\nResults:")
    print("Original data:")
    print(df.describe())
    print("\nSynthetic data:")
    print(synthetic_df.describe())

    return synthetic_df


if __name__ == "__main__":
    # Run the improved example
    synthetic_data = improved_training_example()

    print(f"\n{'='*60}")
    print("To use with your own data:")
    print("1. Replace df with your DataFrame")
    print("2. Use quick_fix_for_your_data(df)")
    print("3. Or copy the improved configuration")
    print(f"{'='*60}")
