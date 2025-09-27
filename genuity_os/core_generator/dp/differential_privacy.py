"""
Differential Privacy Module for Genuity

A minimal module that applies differential privacy to preprocessed DataFrames
with configurable privacy parameters and minimal noise addition.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import laplace, gaussian_kde
import logging

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifferentialPrivacyProcessor:
    """
    Minimal differential privacy processor for preprocessed DataFrames.

    Applies differential privacy with minimal noise addition while preserving
    data utility for synthetic data generation.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        noise_scale: float = 0.1,
        categorical_noise_prob: float = 0.05,
        numerical_noise_std: float = 0.01,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the differential privacy processor.

        Args:
            epsilon: Privacy budget (lower = more private, default: 1.0)
            delta: Privacy parameter (default: 1e-5)
            noise_scale: Scale factor for noise addition (default: 0.1)
            categorical_noise_prob: Probability of flipping categorical values (default: 0.05)
            numerical_noise_std: Standard deviation for numerical noise (default: 0.01)
            random_state: Random seed for reproducibility
        """
        self.epsilon = epsilon
        self.delta = delta
        self.noise_scale = noise_scale
        self.categorical_noise_prob = categorical_noise_prob
        self.numerical_noise_std = numerical_noise_std
        self.random_state = random_state

        if random_state is not None:
            # Use secure random state management
            import random

            np.random.seed(random_state)
            random.seed(random_state)
            # Set PyTorch seed if available
            try:
                import torch

                torch.manual_seed(random_state)
            except ImportError:
                pass

        # Store original data statistics
        self.original_stats = {}
        self.scalers = {}

        logger.info(f"Initialized DP processor with epsilon={epsilon}, delta={delta}")

    def _detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Detect numerical and categorical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        for col in numerical_cols:
            if df[col].nunique() <= min(10, len(df) * 0.1):
                categorical_cols.append(col)
                numerical_cols.remove(col)

        return numerical_cols, categorical_cols

    def _add_laplace_noise(self, data: np.ndarray, sensitivity: float) -> np.ndarray:
        """Add Laplace noise for differential privacy."""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

    def _add_gaussian_noise(self, data: np.ndarray, sensitivity: float) -> np.ndarray:
        """Add Gaussian noise for differential privacy."""
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise

    def _apply_numerical_dp(
        self, df: pd.DataFrame, numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Apply differential privacy to numerical columns."""
        df_dp = df.copy()

        for col in numerical_cols:
            if col not in df.columns:
                continue

            # Get column data
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            # Calculate sensitivity (range of the data)
            data_range = col_data.max() - col_data.min()
            if data_range == 0:
                data_range = 1.0  # Avoid division by zero

            # Scale sensitivity by noise_scale for minimal noise
            sensitivity = data_range * self.noise_scale

            # Add Laplace noise
            noisy_data = self._add_laplace_noise(col_data.values, sensitivity)

            # Replace original values with noisy ones
            df_dp.loc[col_data.index, col] = noisy_data

            # Ensure values stay within reasonable bounds
            df_dp[col] = np.clip(
                df_dp[col],
                col_data.min() - data_range * 0.1,
                col_data.max() + data_range * 0.1,
            )

        return df_dp

    def _apply_categorical_dp(
        self, df: pd.DataFrame, categorical_cols: List[str]
    ) -> pd.DataFrame:
        """Apply differential privacy to categorical columns."""
        df_dp = df.copy()

        for col in categorical_cols:
            if col not in df.columns:
                continue

            # Get unique values and their counts
            value_counts = df[col].value_counts()
            total_count = len(df[col].dropna())

            if total_count == 0:
                continue

            # Calculate sensitivity for categorical data
            sensitivity = 1.0  # Adding/removing one record changes count by at most 1

            # Add noise to value counts
            noisy_counts = {}
            for value, count in value_counts.items():
                # Add Laplace noise to counts
                noisy_count = count + np.random.laplace(0, sensitivity / self.epsilon)
                noisy_counts[value] = max(0, noisy_count)  # Ensure non-negative

            # Normalize to get probabilities
            total_noisy = sum(noisy_counts.values())
            if total_noisy > 0:
                probabilities = {k: v / total_noisy for k, v in noisy_counts.items()}
            else:
                # Fallback to uniform distribution
                probabilities = {
                    k: 1.0 / len(value_counts) for k in value_counts.keys()
                }

            # Sample new values based on noisy probabilities
            values = list(probabilities.keys())
            probs = list(probabilities.values())

            # Normalize probabilities
            probs = np.array(probs) / np.sum(probs)

            # Sample new categorical values
            new_values = np.random.choice(values, size=len(df), p=probs)
            df_dp[col] = new_values

        return df_dp

    def _apply_minimal_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply minimal noise addition for very conservative privacy."""
        df_dp = df.copy()

        # Add very small Gaussian noise to all numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if col not in df.columns:
                continue

            # Add minimal Gaussian noise
            noise = np.random.normal(0, self.numerical_noise_std, len(df))
            df_dp[col] = df[col] + noise

        # Add minimal categorical noise
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        for col in categorical_cols:
            if col not in df.columns:
                continue

            # Randomly flip a small percentage of categorical values
            mask = np.random.random(len(df)) < self.categorical_noise_prob
            if mask.any():
                # Get unique values for this column
                unique_values = df[col].dropna().unique()
                if len(unique_values) > 1:
                    # Randomly assign different values to noisy entries
                    random_values = np.random.choice(unique_values, size=mask.sum())
                    df_dp.loc[mask, col] = random_values

        return df_dp

    def apply_dp(
        self,
        df: pd.DataFrame,
        method: str = "minimal",
        preserve_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply differential privacy to the DataFrame.

        Args:
            df: Input DataFrame (should be preprocessed)
            method: DP method ("minimal", "laplace", "gaussian")
            preserve_columns: Columns to preserve without modification

        Returns:
            DataFrame with differential privacy applied
        """
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return df

        logger.info(
            f"Applying {method} differential privacy to DataFrame of shape {df.shape}"
        )

        # Store original statistics
        self.original_stats = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
        }

        # Detect column types
        numerical_cols, categorical_cols = self._detect_column_types(df)

        # Filter out preserved columns
        if preserve_columns:
            numerical_cols = [
                col for col in numerical_cols if col not in preserve_columns
            ]
            categorical_cols = [
                col for col in categorical_cols if col not in preserve_columns
            ]

        df_dp = df.copy()

        if method == "minimal":
            # Apply minimal noise (default)
            df_dp = self._apply_minimal_noise(df_dp)

        elif method == "laplace":
            # Apply Laplace noise to numerical columns
            df_dp = self._apply_numerical_dp(df_dp, numerical_cols)

            # Apply categorical DP
            df_dp = self._apply_categorical_dp(df_dp, categorical_cols)

        elif method == "gaussian":
            # Apply Gaussian noise (for larger datasets)
            for col in numerical_cols:
                if col not in df_dp.columns:
                    continue

                col_data = df_dp[col].dropna()
                if len(col_data) == 0:
                    continue

                data_range = col_data.max() - col_data.min()
                if data_range == 0:
                    data_range = 1.0

                sensitivity = data_range * self.noise_scale
                noisy_data = self._add_gaussian_noise(col_data.values, sensitivity)
                df_dp.loc[col_data.index, col] = noisy_data

            # Apply categorical DP
            df_dp = self._apply_categorical_dp(df_dp, categorical_cols)

        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'minimal', 'laplace', or 'gaussian'"
            )

        logger.info(f"Applied DP with method '{method}'. Output shape: {df_dp.shape}")

        return df_dp

    def get_privacy_info(self) -> Dict:
        """Get information about the applied privacy parameters."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_scale": self.noise_scale,
            "categorical_noise_prob": self.categorical_noise_prob,
            "numerical_noise_std": self.numerical_noise_std,
            "original_stats": self.original_stats,
        }


# Convenience function for quick DP application
def apply_differential_privacy(
    df: pd.DataFrame,
    epsilon: float = 1.0,
    method: str = "minimal",
    preserve_columns: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Quick function to apply differential privacy to a DataFrame.

    Args:
        df: Input DataFrame (should be preprocessed)
        epsilon: Privacy budget (default: 1.0)
        method: DP method ("minimal", "laplace", "gaussian")
        preserve_columns: Columns to preserve without modification
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with differential privacy applied
    """
    processor = DifferentialPrivacyProcessor(epsilon=epsilon, random_state=random_state)

    return processor.apply_dp(df, method=method, preserve_columns=preserve_columns)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np

    # Create sample preprocessed data
    np.random.seed(42)
    n_samples = 1000

    # Sample preprocessed DataFrame
    df_sample = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, n_samples),
            "feature_2": np.random.normal(0, 1, n_samples),
            "feature_3": np.random.choice([0, 1], n_samples),
            "category_1": np.random.choice(["A", "B", "C"], n_samples),
            "category_2": np.random.choice(["X", "Y"], n_samples),
            "target": np.random.choice([0, 1], n_samples),
        }
    )

    print("Original DataFrame:")
    print(df_sample.head())
    print(f"Shape: {df_sample.shape}")

    # Apply minimal differential privacy
    df_dp = apply_differential_privacy(
        df=df_sample,
        epsilon=1.0,
        method="minimal",
        preserve_columns=["target"],  # Preserve target column
    )

    print("\nAfter applying differential privacy:")
    print(df_dp.head())
    print(f"Shape: {df_dp.shape}")

    # Show privacy information
    processor = DifferentialPrivacyProcessor(epsilon=1.0)
    processor.apply_dp(df_sample, method="minimal")
    privacy_info = processor.get_privacy_info()

    print("\nPrivacy Information:")
    for key, value in privacy_info.items():
        if key != "original_stats":
            print(f"  {key}: {value}")
