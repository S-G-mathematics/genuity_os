import numpy as np
from tvae import TVAEAPI


def test_tvae_basic():
    data = np.random.randn(100, 4)
    continuous_cols = [0, 1]
    categorical_cols = [2, 3]
    tvae = TVAEAPI(model_type="basic")
    tvae.fit(data, continuous_cols, categorical_cols, epochs=2)
    samples = tvae.generate(10)
    assert samples.shape[0] == 10
    assert samples.shape[1] == 4
