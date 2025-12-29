from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - depends on host torch install
    TORCH_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]
else:
    TORCH_IMPORT_ERROR = None


@pytest.fixture(scope="session")
def torch_available() -> bool:
    if TORCH_IMPORT_ERROR is not None:
        pytest.skip(f"Torch unavailable: {TORCH_IMPORT_ERROR}")
    return True


@pytest.fixture(scope="session")
def trained_estimator(torch_available: bool):
    from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig  # noqa: PLC0415

    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 3))
    noise = (0.2 + 0.3 * np.abs(X[:, 0])) * rng.normal(size=X.shape[0])
    y = 0.5 * X[:, 0] - 0.25 * X[:, 1] + 0.1 * X[:, 2] + noise

    grid = np.linspace(y.min() - 1.0, y.max() + 1.0, 64)

    config = CondensiteTorchCDEConfig(
        epochs=2,
        patience=2,
        batch_size=32,
        m_aux=24,
        hidden_sizes=(32, 32),
        bandwidth=0.12,
        lr=5e-3,
        sampler="stratified",
        amp=False,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=7).fit(X, y)
    return estimator, X, y, grid
