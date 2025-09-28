# Genuity Open Strategy

A comprehensive synthetic tabular data generation framework with multiple state-of-the-art algorithms, advanced data preprocessing, and differential privacy support.

## Features

- **Multiple Generation Algorithms**: CTGAN, TabuDiff, and TVAE implementations
- **Advanced Data Preprocessing**: Automatic column classification, encoding, scaling, and PCA
- **Differential Privacy**: Built-in privacy protection with configurable parameters
- **High-Level APIs**: Easy-to-use interfaces for all components
- **Production Ready**: Comprehensive error handling, logging, and device management

## Installation

### For Kaggle Users

Kaggle supports installing packages directly from GitHub. Add this to your Kaggle notebook:

```python
# Install Genuity OS in Kaggle
!pip install git+https://github.com/S-G-mathematics/genuity_os.git
```

### Kaggle Notebook Example

```python
# Cell 1: Install the package
!pip install git+https://github.com/yourusername/genuity_os.git

# Cell 2: Import and use
import pandas as pd
import numpy as np
from genuity_os.core_generator.ctgan.ctgan import CTGANAPI

# Your code here...
```

## Usage

### Quick Start

```python
import pandas as pd
import numpy as np
from genuity_os.core_generator.ctgan.ctgan.utils.api import CTGANAPI

# Create sample data
data = np.random.randn(1000, 5)
continuous_cols = [0, 1, 2]
categorical_cols = [3, 4]

# Train CTGAN model
api = CTGANAPI()
losses = api.fit(
    data=data,
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols,
    epochs=1000
)

# Generate synthetic data
synthetic_data = api.generate(1000)
print(synthetic_data.shape)
```

### Data Preprocessing

```python
from genuity_os.data_processor.data_preprocess import TabularPreprocessor

# Initialize preprocessor
preprocessor = TabularPreprocessor(
    scaler_type="standard",
    encoding_strategy="onehot",
)

# Fit and transform data
result = preprocessor.fit_transform(df)
preprocessed_data = result["preprocessed"]

# Save preprocessor state
preprocessor.save_preprocessor("preprocessor.joblib")
```

### TabuDiff Usage

```python
from genuity_os.core_generator.tabudiff.tabudiff.utils.api import TabuDiffAPI
from genuity_os.core_generator.tabudiff.tabudiff.config.config import TabuDiffConfig

# Configure TabuDiff
config = TabuDiffConfig(
    learning_rate=2e-4,
    batch_size=128,
    num_epochs=200,
    num_diffusion_steps=1000,
    hidden_dim=512
)

# Train and generate
api = TabuDiffAPI(config)
api.fit_dataframe(df)
synthetic_df = api.generate_dataframe(num_samples=1000)
```

### TVAE Usage

```python
from genuity_os.core_generator.tvae.tvae.utils.api import TVAEAPI

# Train TVAE model
tvae = TVAEAPI(model_type="basic")
losses = tvae.fit(
    data=data,
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols,
    epochs=1000
)

# Generate samples
samples = tvae.generate(1000)
```

### Differential Privacy

```python
from genuity_os.core_generator.dp.differential_privacy import DifferentialPrivacyProcessor

# Apply differential privacy
dp_processor = DifferentialPrivacyProcessor(
    epsilon=1.0,
    delta=1e-5,
    noise_scale=0.1
)

# Apply DP to preprocessed data
dp_data = dp_processor.apply_dp(preprocessed_data, method="minimal")
```

## Examples

### Complete Pipeline Example

```python
import pandas as pd
import numpy as np
from genuity_os.data_processor.data_preprocess import TabularPreprocessor
from genuity_os.core_generator.ctgan.ctgan.utils.api import CTGANAPI
from genuity_os.core_generator.dp.differential_privacy import DifferentialPrivacyProcessor

# 1. Load and preprocess data
preprocessor = TabularPreprocessor(
    scaler_type="standard",
    encoding_strategy="onehot",
    normalize_data=True
)

result = preprocessor.fit_transform(df)
preprocessed_data = result["preprocessed"]

# 2. Apply differential privacy (optional)
dp_processor = DifferentialPrivacyProcessor(epsilon=1.0)
dp_data = dp_processor.apply_dp(preprocessed_data, method="minimal")

# 3. Train synthetic data generator
api = CTGANAPI()
losses = api.fit(
    data=dp_data.values,
    continuous_cols=list(range(len(result["continuous"].columns))),
    categorical_cols=list(range(len(result["categorical"].columns))),
    epochs=1000
)

# 4. Generate synthetic data
synthetic_data = api.generate(1000)

# 5. Save model
api.save("ctgan_model.pth")
```

## Modules

### Core Generators

#### CTGAN (Conditional Tabular GAN)
- **Location**: `genuity_os.core_generator.ctgan`
- **API**: `CTGANAPI`
- **Features**: Improved stability with spectral normalization, gradient penalty, and diversity loss
- **Configuration**: `CTGANConfig`

#### TabuDiff (Tabular Diffusion)
- **Location**: `genuity_os.core_generator.tabudiff`
- **API**: `TabuDiffAPI`
- **Features**: Diffusion-based generation with configurable noise schedules
- **Configuration**: `TabuDiffConfig`

#### TVAE (Tabular Variational Autoencoder)
- **Location**: `genuity_os.core_generator.tvae`
- **API**: `TVAEAPI`
- **Features**: Variational autoencoder with β-VAE support
- **Configuration**: `TVAEConfig`

### Data Processing

#### TabularPreprocessor
- **Location**: `genuity_os.data_processor.data_preprocess`
- **Features**:
  - Automatic column classification (continuous, categorical, binary)
  - Multiple encoding strategies (onehot, ordinal, binary, hash, frequency, target)
  - Scaling options (standard, minmax, robust, maxabs)
  - PCA dimensionality reduction
  - Outlier detection and flagging
  - Missing value imputation (mean, median, mode, ML-based)

#### TabularPostprocessor
- **Location**: `genuity_os.data_processor.data_postprocess`
- **Features**:
  - Inverse transformations
  - Data reconstruction
  - New data transformation using saved preprocessor state

### Differential Privacy

#### DifferentialPrivacyProcessor
- **Location**: `genuity_os.core_generator.dp.differential_privacy`
- **Features**:
  - Multiple noise mechanisms (Laplace, Gaussian, minimal)
  - Configurable privacy parameters (epsilon, delta)
  - Column-specific privacy preservation
  - Privacy budget management

### Utilities

#### Device Management
- **Location**: `genuity_os.utils.device`
- **Features**: Automatic device detection (CUDA, MPS, CPU), memory management

#### Logging
- **Location**: `genuity_os.utils.logger`
- **Features**: Structured logging with performance monitoring and operation tracking

## Configuration

All modules support extensive configuration through dataclass-based config objects:

- `CTGANConfig`: Generator/discriminator architecture, training parameters
- `TabuDiffConfig`: Diffusion parameters, model architecture, training settings
- `TVAEConfig`: VAE architecture, β-VAE parameters, advanced features
- `TabularPreprocessor`: Preprocessing pipeline configuration

## Security Features

- **API-Only Access**: No console commands are exposed - the library can only be used through Python imports
- **Input Validation**: Comprehensive validation prevents malicious input
- **File Size Limits**: Model files are limited to 100MB to prevent resource exhaustion
- **Secure Loading**: Model loading uses `weights_only=True` to prevent code execution
- **Differential Privacy**: Built-in privacy protection for sensitive data

## Advanced Features

### Model Persistence
All generators support model saving and loading:

```python
# Save trained model
api.save("model.pth")

# Load saved model
api.load("model.pth")
```

### Performance Monitoring
Built-in logging and performance tracking:

```python
from genuity_os.utils.logger import GenuityLogger

logger = GenuityLogger()
logger.start_operation("Training CTGAN")
# ... training code ...
logger.end_operation(success=True)
```

### Device Management
Automatic device detection and management:

```python
from genuity_os.utils.device import get_device, get_device_info

device = get_device()
info = get_device_info()
print(f"Using device: {info['device']}")
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Scikit-learn
- Category Encoders
- SciPy
- Joblib

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Genuity OS in your research, please cite:

```bibtex
@software{genuity_os,
  title={Genuity OS: A Comprehensive Synthetic Tabular Data Generation Framework},
  author={Genuity IO},
  year={2025},
  url={https://github.com/yourusername/genuity_os}
}
```
