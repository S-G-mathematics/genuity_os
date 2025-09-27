#!/usr/bin/env python3
"""
Kaggle Example for Genuity OS
This example demonstrates how to use Genuity OS in Kaggle notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Install Genuity OS (run this in the first cell of your Kaggle notebook)
# !pip install git+https://github.com/yourusername/genuity_os.git

# Import Genuity OS modules
from genuity_os.data_processor.data_preprocess import TabularPreprocessor
from genuity_os.core_generator.ctgan.ctgan.utils.api import CTGANAPI
from genuity_os.core_generator.tabudiff.tabudiff.utils.api import TabuDiffAPI
from genuity_os.core_generator.tvae.tvae.utils.api import TVAEAPI
from genuity_os.core_generator.dp.differential_privacy import (
    DifferentialPrivacyProcessor,
)
from genuity_os.utils.logger import GenuityLogger


def create_sample_data():
    """Create sample tabular data for demonstration."""
    np.random.seed(42)

    # Create sample data with mixed column types
    n_samples = 1000

    # Continuous features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.lognormal(10, 0.5, n_samples)
    score = np.random.uniform(0, 100, n_samples)

    # Categorical features
    education = np.random.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        n_samples,
        p=[0.3, 0.4, 0.25, 0.05],
    )
    city = np.random.choice(
        ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], n_samples
    )
    gender = np.random.choice(["Male", "Female"], n_samples)

    # Binary feature
    has_insurance = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    # Create DataFrame
    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "score": score,
            "education": education,
            "city": city,
            "gender": gender,
            "has_insurance": has_insurance,
        }
    )

    return df


def demonstrate_preprocessing(df):
    """Demonstrate data preprocessing with Genuity OS."""
    print("=== Data Preprocessing Demo ===")
    print(f"Original data shape: {df.shape}")
    print(f"Original data types:\n{df.dtypes}")

    # Initialize preprocessor
    preprocessor = TabularPreprocessor(
        scaler_type="standard",
        encoding_strategy="onehot",
        normalize_data=True,
        handle_outliers=True,
        apply_pca=False,
    )

    # Fit and transform data
    result = preprocessor.fit_transform(df)
    preprocessed_data = result["preprocessed"]

    print(f"\nPreprocessed data shape: {preprocessed_data.shape}")
    print(f"Continuous columns: {len(result['continuous'].columns)}")
    print(f"Categorical columns: {len(result['categorical'].columns)}")

    return preprocessed_data, result


def demonstrate_ctgan(data, continuous_cols, categorical_cols):
    """Demonstrate CTGAN synthetic data generation."""
    print("\n=== CTGAN Demo ===")

    # Initialize CTGAN API
    api = CTGANAPI()

    # Train the model
    print("Training CTGAN model...")
    losses = api.fit(
        data=data.values,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        epochs=100,  # Reduced for demo
    )

    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = api.generate(1000)

    print(f"Generated synthetic data shape: {synthetic_data.shape}")
    return synthetic_data, losses


def demonstrate_tabudiff(df):
    """Demonstrate TabuDiff synthetic data generation."""
    print("\n=== TabuDiff Demo ===")

    # Initialize TabuDiff API with minimal config for demo
    from genuity_os.core_generator.tabudiff.tabudiff.config.config import TabuDiffConfig

    config = TabuDiffConfig(
        learning_rate=2e-4,
        batch_size=64,
        num_epochs=50,  # Reduced for demo
        num_diffusion_steps=100,  # Reduced for demo
        hidden_dim=256,
    )

    api = TabuDiffAPI(config)

    # Train and generate
    print("Training TabuDiff model...")
    api.fit_dataframe(df)

    print("Generating synthetic data...")
    synthetic_df = api.generate_dataframe(num_samples=1000)

    print(f"Generated synthetic data shape: {synthetic_df.shape}")
    return synthetic_df


def demonstrate_tvae(data, continuous_cols, categorical_cols):
    """Demonstrate TVAE synthetic data generation."""
    print("\n=== TVAE Demo ===")

    # Initialize TVAE API
    tvae = TVAEAPI(model_type="basic")

    # Train the model
    print("Training TVAE model...")
    losses = tvae.fit(
        data=data.values,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        epochs=100,  # Reduced for demo
    )

    # Generate synthetic data
    print("Generating synthetic data...")
    samples = tvae.generate(1000)

    print(f"Generated synthetic data shape: {samples.shape}")
    return samples, losses


def demonstrate_differential_privacy(data):
    """Demonstrate differential privacy."""
    print("\n=== Differential Privacy Demo ===")

    # Apply differential privacy
    dp_processor = DifferentialPrivacyProcessor(
        epsilon=1.0, delta=1e-5, noise_scale=0.1
    )

    # Apply DP to data
    dp_data = dp_processor.apply_dp(data, method="minimal")

    print(f"Original data shape: {data.shape}")
    print(f"DP data shape: {dp_data.shape}")
    print("Differential privacy applied successfully!")

    return dp_data


def visualize_comparison(original_df, synthetic_data, continuous_cols):
    """Visualize comparison between original and synthetic data."""
    print("\n=== Visualization ===")

    # Convert synthetic data back to DataFrame for easier plotting
    if isinstance(synthetic_data, np.ndarray):
        # Get column names for continuous features
        continuous_col_names = [f"feature_{i}" for i in continuous_cols]
        synthetic_df = pd.DataFrame(
            synthetic_data[:, continuous_cols], columns=continuous_col_names
        )

        # Plot comparison for first few continuous features
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for i, col in enumerate(continuous_col_names[:4]):
            row = i // 2
            col_idx = i % 2

            axes[row, col_idx].hist(
                original_df.iloc[:, continuous_cols[i]],
                alpha=0.5,
                label="Original",
                bins=30,
            )
            axes[row, col_idx].hist(
                synthetic_df[col], alpha=0.5, label="Synthetic", bins=30
            )
            axes[row, col_idx].set_title(f"Feature {i+1} Distribution")
            axes[row, col_idx].legend()

        plt.tight_layout()
        plt.show()


def main():
    """Main demonstration function."""
    print("🚀 Genuity OS Kaggle Demo")
    print("=" * 50)

    # Create sample data
    df = create_sample_data()
    print(
        f"Created sample dataset with {df.shape[0]} samples and {df.shape[1]} features"
    )

    # Demonstrate preprocessing
    preprocessed_data, result = demonstrate_preprocessing(df)

    # Get column indices for continuous features
    continuous_cols = list(range(len(result["continuous"].columns)))
    categorical_cols = list(
        range(
            len(result["continuous"].columns),
            len(result["continuous"].columns) + len(result["categorical"].columns),
        )
    )

    # Demonstrate CTGAN
    ctgan_synthetic, ctgan_losses = demonstrate_ctgan(
        preprocessed_data, continuous_cols, categorical_cols
    )

    # Demonstrate TabuDiff (on original DataFrame)
    tabudiff_synthetic = demonstrate_tabudiff(df)

    # Demonstrate TVAE
    tvae_synthetic, tvae_losses = demonstrate_tvae(
        preprocessed_data, continuous_cols, categorical_cols
    )

    # Demonstrate Differential Privacy
    dp_data = demonstrate_differential_privacy(preprocessed_data)

    # Visualize results
    visualize_comparison(df, ctgan_synthetic, continuous_cols[:4])

    print("\n✅ Demo completed successfully!")
    print("\n📊 Summary:")
    print(f"- Original data: {df.shape}")
    print(f"- CTGAN synthetic: {ctgan_synthetic.shape}")
    print(f"- TabuDiff synthetic: {tabudiff_synthetic.shape}")
    print(f"- TVAE synthetic: {tvae_synthetic.shape}")
    print(f"- DP processed: {dp_data.shape}")


if __name__ == "__main__":
    main()
