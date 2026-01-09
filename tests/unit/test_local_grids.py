from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_cde.grids import make_y_grid
from condensite_torch import (
    CondensiteTorchCDE,
    CondensiteTorchCDEConfig,
    local_grids,
    make_local_grid,
)

pytestmark = pytest.mark.unit
_CACHE_EXPECTED_REBUILDS = 2


def _make_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(128, 2))
    y = (0.3 + 0.2 * np.abs(X[:, 0])) * rng.normal(size=X.shape[0]) + 0.5 * X[:, 0] - 0.1 * X[:, 1]
    return X, y


def test_make_local_grid_returns_sorted_rows(trained_estimator) -> None:
    estimator, X, _y, _grid = trained_estimator
    grids = make_local_grid(estimator, X[:4], grid_size=32)
    assert grids.shape == (4, 32)
    assert np.all(np.diff(grids, axis=1) > 0)


def test_local_grids_cover_targets_better_than_global() -> None:
    X, y = _make_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(24, 24),
        m_aux=20,
        epochs=4,
        patience=2,
        sampler="sobol",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=6).fit(X, y)
    split = 96
    X_test = X[split:]
    y_test = y[split:]
    global_grid = make_y_grid(y[:split], grid_size=64)
    global_in = np.mean((y_test >= global_grid[0]) & (y_test <= global_grid[-1]))
    local_grids = make_local_grid(estimator, X_test, grid_size=64)
    local_in = np.mean(
        (y_test >= local_grids[:, 0]) & (y_test <= local_grids[:, -1]),
    )
    assert local_in >= global_in
    preview = 5
    grid_size = 64
    pdf_local = estimator.predict_density(X_test[:preview], local_grids[:preview])
    assert pdf_local.shape == (preview, grid_size)
    pdf_flag = estimator.predict_density(X_test[:preview], None, use_local_grid=True)
    assert pdf_flag.shape[0] == preview
    assert pdf_local.shape == (preview, grid_size)


def test_local_grid_cache_reused(monkeypatch, trained_estimator) -> None:
    estimator, X, _y, _grid = trained_estimator
    call_count = 0
    original = local_grids.make_local_grid

    def wrapped(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(local_grids, "make_local_grid", wrapped)
    estimator.predict_density(X[:8], None, use_local_grid=True)
    estimator.predict_density(X[:8], None, use_local_grid=True)
    assert call_count == 1
    estimator.predict_density(
        X[:8],
        None,
        use_local_grid=True,
        local_grid_params={"padding": 0.2},
    )
    assert call_count == _CACHE_EXPECTED_REBUILDS
