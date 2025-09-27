import numpy as np
from tvae import TVAEAPI

data = np.random.randn(1000, 5)
continuous_cols = [0, 1, 2]
categorical_cols = [3, 4]

tvae = TVAEAPI(model_type="basic")
losses = tvae.fit(
    data=data,
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols,
    epochs=1000,
)
samples = tvae.generate(1000)
print(samples)
