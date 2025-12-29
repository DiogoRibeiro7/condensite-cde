from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig
from condensite_torch.metrics import crps_from_cdf, nll_from_pdf

pytestmark = pytest.mark.regression


def _make_dataset(n_samples: int = 192) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, 2))
    y = 0.4 * np.sin(X[:, 0]) - 0.25 * X[:, 1] + 0.15 * rng.normal(size=n_samples)
    split = int(0.75 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def test_density_snapshot_matches_baseline() -> None:
    base = Path("tests/regression/baselines")
    metrics_path = base / "baseline_metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not data.get("generated"):
        pytest.skip("Regression baselines not generated; run scripts/generate_regression_baselines.py")
    bundle = np.load(base / "baseline_pdf.npz")
    pdf_expected = bundle["pdf_expected"].astype(np.float64)
    grid = bundle["y_grid"].astype(np.float64)

    X_train, y_train, X_test, y_test = _make_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=24,
        epochs=5,
        patience=2,
        sampler="sobol",
        bandwidth=0.12,
        normalization_lambda=0.1,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=11).fit(X_train, y_train)
    pdf = estimator.predict_density(X_test, grid)
    np.testing.assert_allclose(pdf, pdf_expected, rtol=1e-2, atol=1e-3)
    cdf = estimator.predict_cdf(X_test, grid)
    nll = nll_from_pdf(y_test, grid, pdf)
    crps = crps_from_cdf(y_test, grid, cdf)
    integral = np.mean(np.trapezoid(pdf, x=grid, axis=1))
    assert abs(nll - data["nll_expected"]) <= 5e-3
    assert abs(crps - data["crps_expected"]) <= 5e-3
    assert abs(integral - data["integral_mean_expected"]) <= 5e-3
