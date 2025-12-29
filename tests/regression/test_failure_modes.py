from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig
from condensite_torch.metrics import nll_from_pdf

pytestmark = pytest.mark.regression


def _dataset(n_samples: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 2))
    noise = (0.1 + 0.2 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.5 * np.sin(X[:, 0]) - 0.3 * X[:, 1] + noise
    split = int(0.75 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def _config() -> CondensiteTorchCDEConfig:
    return CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=20,
        epochs=4,
        patience=2,
        sampler="sobol",
        bandwidth=0.12,
        normalization_lambda=0.1,
        positive_output=True,
    )


def test_ignoring_yprime_worsens_nll(monkeypatch: pytest.MonkeyPatch) -> None:
    X_train, y_train, X_test, y_test = _dataset()
    grid = make_y_grid(y_train, grid_size=64, mode="quantile")
    baseline = CondensiteTorchCDE(config=_config(), random_seed=3).fit(X_train, y_train)
    pdf_ref = baseline.predict_density(X_test, grid)
    nll_ref = nll_from_pdf(y_test, grid, pdf_ref)

    original = CondensiteTorchCDE._prepare_training_batch

    def fake_prepare(self, X_batch, y_batch):
        features, targets, weights = original(self, X_batch, y_batch)
        features = features.clone()
        features[:, -1] = 0.0
        return features, targets, weights

    monkeypatch.setattr(CondensiteTorchCDE, "_prepare_training_batch", fake_prepare)
    degraded = CondensiteTorchCDE(config=_config(), random_seed=3).fit(X_train, y_train)
    pdf_bad = degraded.predict_density(X_test, grid)
    nll_bad = nll_from_pdf(y_test, grid, pdf_bad)
    assert nll_bad > nll_ref + 0.05


def test_disabling_normalization_breaks_pdf_integrals(monkeypatch: pytest.MonkeyPatch) -> None:
    X_train, y_train, X_test, _ = _dataset()
    grid = make_y_grid(y_train, grid_size=64, mode="quantile")
    estimator = CondensiteTorchCDE(config=_config(), random_seed=5).fit(X_train, y_train)
    original = CondensiteTorchCDE._predict_density_internal

    def fake_predict(self, X, y_grid, *, head=None):
        pdf = original(self, X, y_grid, head=head)
        return pdf * 2.0

    monkeypatch.setattr(CondensiteTorchCDE, "_predict_density_internal", fake_predict)
    pdf = estimator.predict_density(X_test, grid)
    mass = np.trapzoid(pdf, x=grid, axis=1)
    assert np.any(mass > 1.5)


def test_invalid_bandwidth_raises() -> None:
    config = _config()
    config.bandwidth = 0.0
    estimator = CondensiteTorchCDE(config=config, random_seed=1)
    X_train, y_train, _, _ = _dataset()
    with pytest.raises(ValueError):
        estimator.fit(X_train, y_train)
