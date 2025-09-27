import pandas as pd
import numpy as np
from tabudiff.basic.utils import TabuDiffAPI
from tabudiff.basic.config import TabuDiffConfig


def main():
    # Create a more realistic dataset with better structure
    np.random.seed(42)
    n_samples = 1000

    # Generate correlated data
    age = np.random.normal(35, 10, n_samples)
    income = 20000 + age * 1000 + np.random.normal(0, 5000, n_samples)
    score = np.random.uniform(0, 100, n_samples)

    df = pd.DataFrame({"age": age, "income": income, "score": score})

    print("Original data statistics:")
    print(df.describe())
    print(f"\nOriginal data shape: {df.shape}")

    # Use improved configuration
    config = TabuDiffConfig(
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=100,  # More epochs for better training
        num_diffusion_steps=1000,
        hidden_dim=256,
        num_hidden_layers=3,
        dropout=0.1,
        normalize_data=True,
        clip_gradients=True,
    )

    # Train the model
    api = TabuDiffAPI(config)
    api.fit_dataframe(df)

    # Generate synthetic data
    synthetic_df = api.generate_dataframe(num_samples=500)

    print("\nSynthetic data statistics:")
    print(synthetic_df.describe())
    print(f"\nSynthetic data shape: {synthetic_df.shape}")

    # Compare correlations
    print("\nOriginal data correlations:")
    print(df.corr())
    print("\nSynthetic data correlations:")
    print(synthetic_df.corr())


if __name__ == "__main__":
    main()
