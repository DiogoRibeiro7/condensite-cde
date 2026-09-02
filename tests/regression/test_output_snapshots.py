from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig
from condensite_torch.diagnostics import coverage_rate, pit_values
from condensite_torch.metrics import crps_from_cdf, nll_from_pdf

pytestmark = pytest.mark.regression

BASELINE_DIR = Path("tests/regression/baselines")
DTYPE_MAP = {"float32": np.float32, "float64": np.float64}
INTERVAL_BOUNDS = {
    "50": (0.25, 0.75),
    "80": (0.1, 0.9),
    "90": (0.05, 0.95),
    "95": (0.025, 0.975),
}
ARRAY_RTOL = 3e-3
ARRAY_ATOL = 5e-4
METRIC_TOL = 2e-3


def _make_dataset(n_samples: int = 192) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, 2))
    y = 0.4 * np.sin(X[:, 0]) - 0.25 * X[:, 1] + 0.15 * rng.normal(size=n_samples)
    split = int(0.75 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def _load_baseline_metadata() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    metrics_path = BASELINE_DIR / "baseline_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not payload.get("generated"):
        pytest.skip(
            "Regression baselines not generated; run scripts/generate_regression_baselines.py",
        )
    quantile_probs = np.asarray(payload["quantile_probs"], dtype=np.float64)
    tail_thresholds = np.asarray(payload["tail_prob_thresholds"], dtype=np.float64)
    pit_edges = np.asarray(payload["pit_bin_edges"], dtype=np.float64)
    return payload["snapshots"], quantile_probs, tail_thresholds, pit_edges


def _tail_matrix(
    estimator: CondensiteTorchCDE,
    X_cast: np.ndarray,
    grid_cast: np.ndarray,
    *,
    thresholds: np.ndarray,
    side: str,
) -> np.ndarray:
    columns = [
        estimator.predict_tail_prob(X_cast, float(th), side=side, y_grid=grid_cast).reshape(-1, 1)
        for th in thresholds
    ]
    return np.hstack(columns)


def _assert_metrics(  # noqa: PLR0913
    *,
    metrics_expected: dict,
    pdf: np.ndarray,
    cdf: np.ndarray,
    y_test: np.ndarray,
    grid_float: np.ndarray,
    pit_edges: np.ndarray,
    quantiles: np.ndarray,
    quantile_probs: np.ndarray,
) -> None:
    nll = float(nll_from_pdf(y_test, grid_float, pdf))
    crps = float(crps_from_cdf(y_test, grid_float, cdf))
    integral = float(np.mean(np.trapezoid(pdf, x=grid_float, axis=1)))
    assert abs(nll - metrics_expected["nll"]) <= METRIC_TOL
    assert abs(crps - metrics_expected["crps"]) <= METRIC_TOL
    assert abs(integral - metrics_expected["integral_mean"]) <= METRIC_TOL

    counts, edges = np.histogram(pit_values(y_test, grid_float, cdf), bins=pit_edges)
    np.testing.assert_array_equal(edges, pit_edges)
    np.testing.assert_array_equal(counts.tolist(), metrics_expected["pit_hist_counts"])

    prob_index = {float(prob): idx for idx, prob in enumerate(quantile_probs)}
    for name, bounds in INTERVAL_BOUNDS.items():
        expected_value = metrics_expected["coverage"][name]
        observed = coverage_rate(
            y_test,
            quantiles[:, prob_index[bounds[0]]],
            quantiles[:, prob_index[bounds[1]]],
        )
        assert abs(observed - expected_value) <= METRIC_TOL


def test_density_snapshot_matches_baseline() -> None:  # noqa: PLR0914
    snapshots, quantile_probs, tail_thresholds, pit_edges = _load_baseline_metadata()
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

    for dtype_name, np_dtype in DTYPE_MAP.items():
        bundle = np.load(BASELINE_DIR / f"baseline_{dtype_name}.npz")
        pdf_expected = bundle["pdf"].astype(np.float64)
        cdf_expected = bundle["cdf"].astype(np.float64)
        grid = bundle["y_grid"].astype(np.float64)
        quantiles_expected = bundle["quantiles"].astype(np.float64)
        tail_right_expected = bundle["tail_right"].astype(np.float64)
        tail_left_expected = bundle["tail_left"].astype(np.float64)

        X_cast = X_test.astype(np_dtype)
        grid_cast = grid.astype(np_dtype)
        pdf = estimator.predict_density(X_cast, grid_cast)
        np.testing.assert_allclose(pdf, pdf_expected, rtol=ARRAY_RTOL, atol=ARRAY_ATOL)
        cdf = estimator.predict_cdf(X_cast, grid_cast)
        np.testing.assert_allclose(cdf, cdf_expected, rtol=ARRAY_RTOL, atol=ARRAY_ATOL)

        quantiles = estimator.predict_quantile(X_cast, quantile_probs, y_grid=grid_cast)
        np.testing.assert_allclose(quantiles, quantiles_expected, rtol=ARRAY_RTOL, atol=ARRAY_ATOL)

        tail_right = _tail_matrix(
            estimator,
            X_cast,
            grid_cast,
            thresholds=tail_thresholds,
            side="right",
        )
        tail_left = _tail_matrix(
            estimator,
            X_cast,
            grid_cast,
            thresholds=tail_thresholds,
            side="left",
        )
        np.testing.assert_allclose(
            tail_right,
            tail_right_expected,
            rtol=ARRAY_RTOL,
            atol=ARRAY_ATOL,
        )
        np.testing.assert_allclose(
            tail_left,
            tail_left_expected,
            rtol=ARRAY_RTOL,
            atol=ARRAY_ATOL,
        )

        grid_float = grid.astype(np.float64, copy=False)
        _assert_metrics(
            metrics_expected=snapshots[dtype_name],
            pdf=pdf,
            cdf=cdf,
            y_test=y_test,
            grid_float=grid_float,
            pit_edges=pit_edges,
            quantiles=quantiles,
            quantile_probs=quantile_probs,
        )
