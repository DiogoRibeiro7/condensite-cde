from __future__ import annotations

import numpy as np
import pytest

from condensite_cde import PandasCondensiteAdapter, SklearnCondensiteRegressor
from condensite_torch import CondensiteTorchCDEConfig

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency missing
    pd = None  # type: ignore[assignment]


@pytest.fixture()
def toy_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3))
    y = 0.5 * X[:, 0] - 0.25 * X[:, 1] + 0.1 * rng.normal(size=64)
    return X, y


@pytest.fixture()
def fast_config():
    return CondensiteTorchCDEConfig(
        epochs=3,
        patience=2,
        batch_size=32,
        m_aux=12,
        hidden_sizes=(16, 16),
        lr=5e-3,
        sampler="sobol",
        amp=False,
    )


def test_sklearn_wrapper_fit_predict_score(toy_data, fast_config, torch_available):
    X, y = toy_data
    adapter = SklearnCondensiteRegressor(
        config=fast_config,
        random_seed=4,
        prediction_strategy="median",
        grid_size=64,
    )
    adapter.fit(X, y)
    preds = adapter.predict(X[:5])
    assert preds.shape == (5,)
    pdf = adapter.predict_density(X[:3], None)
    assert pdf.shape == (3, adapter._prediction_grid.size)
    interval = adapter.predict_interval(X[:2], coverage=0.8)
    assert interval[0].shape == interval[1].shape == (2,)
    score = adapter.score(X[:10], y[:10])
    assert np.isfinite(score)


def test_sklearn_wrapper_save_load_roundtrip(tmp_path, toy_data, fast_config, torch_available):
    X, y = toy_data
    adapter = SklearnCondensiteRegressor(
        config=fast_config,
        random_seed=5,
        grid_size=32,
    ).fit(X, y)
    save_dir = tmp_path / "bundle"
    adapter.save(save_dir)
    restored = SklearnCondensiteRegressor.load(save_dir, map_location="cpu")
    pdf_a = adapter.predict_density(X[:2], adapter._prediction_grid)
    pdf_b = restored.predict_density(X[:2], restored._prediction_grid)
    assert np.allclose(pdf_a, pdf_b, atol=1e-5)


@pytest.mark.skipif(pd is None, reason="pandas optional dependency")
def test_pandas_adapter_flow(
    toy_data, fast_config, torch_available
):  # pragma: no cover - requires pandas

    X, y = toy_data
    frame = pd.DataFrame(X, columns=["feat0", "feat1", "feat2"])
    frame["target"] = y
    adapter = PandasCondensiteAdapter(config=fast_config, random_seed=3, grid_size=32)
    adapter.fit(frame, target="target")
    preds = adapter.predict(frame.head(3))
    assert list(preds.index) == list(frame.head(3).index)
    density_frame = frame.head(2)
    pdf_df = adapter.predict_density(density_frame)
    assert pdf_df.shape[0] == len(density_frame)
    interval_df = adapter.predict_interval(frame.head(4))
    assert list(interval_df.columns) == ["low", "high"]
