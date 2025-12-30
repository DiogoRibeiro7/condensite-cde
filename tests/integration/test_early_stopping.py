from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on runner environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig, nll_from_pdf

pytestmark = pytest.mark.integration


def _dataset(n_samples: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 2))
    noise = (0.2 + 0.1 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.5 * X[:, 0] - 0.25 * X[:, 1] + noise
    return X, y


def test_early_stopping_triggers_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _dataset()
    split = int(0.7 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    original_nll = nll_from_pdf
    fake_values = iter([0.6, 0.5, 0.55, 0.7, 0.72])

    def fake_nll(y_true, grid, pdf):
        value = next(fake_values, 0.72)
        return value

    monkeypatch.setattr("condensite_torch.estimator.nll_from_pdf", fake_nll)

    config = CondensiteTorchCDEConfig(
        hidden_sizes=(16, 16),
        m_aux=12,
        epochs=10,
        patience=1,
        batch_size=16,
        sampler="sobol",
        monitor_metric="val_nll",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=4)
    estimator.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    history = estimator.training_history
    assert history, "expected at least one epoch"
    assert len(history) < config.epochs
    assert estimator._best_epoch < len(history) - 1
    assert estimator._restored_best_epoch == estimator._best_epoch
    assert all("val_integral_error" in record for record in history)

    # After monkeypatch teardown, evaluate real validation metric for sanity.
    grid = make_y_grid(y_train, grid_size=80, mode="quantile")
    pdf = estimator.predict_density(X_val, grid)
    val_nll_actual = original_nll(y_val, grid, pdf)
    assert np.isfinite(val_nll_actual)
