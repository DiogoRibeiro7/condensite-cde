from __future__ import annotations

from types import MethodType

import numpy as np
import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - depends on runner environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

import condensite_torch.estimator as estimator_module
from condensite_torch import (
    CondensiteTorchCDE,
    CondensiteTorchCDEConfig,
    crps_from_cdf,
    nll_from_pdf,
    sample_yprime,
)
from condensite_torch.aux_sampling import ImportanceSampler

CDF_MONOTONIC_TOL = 1e-4
MONOTONIC_TOL = 1e-6


def test_importance_sampler_is_deterministic() -> None:
    arr = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    sampler = ImportanceSampler.from_array(arr, bins=8, tail_bonus=0.1)
    samples_a, weights_a = sampler.draw((2, 3), seed=123)
    samples_b, weights_b = sampler.draw((2, 3), seed=123)
    samples_c, _ = sampler.draw((2, 3), seed=321)
    assert torch.allclose(samples_a, samples_b)
    assert torch.allclose(weights_a, weights_b)
    assert not torch.allclose(samples_a, samples_c)
    assert torch.all(weights_a > 0)


def test_sample_yprime_is_deterministic_with_seed() -> None:
    a = sample_yprime("stratified", (2, 3), seed=123)
    b = sample_yprime("stratified", (2, 3), seed=123)
    c = sample_yprime("stratified", (2, 3), seed=321)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_condensite_training_produces_valid_pdf_and_cdf() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 2))
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * rng.normal(size=64)
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=8,
        epochs=6,
        patience=3,
        batch_size=16,
        lr=5e-3,
        bandwidth=0.2,
        sampler="sobol",
        amp=False,
        positive_output=True,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=7)
    estimator.fit(X, y)
    grid = np.linspace(y.min() - 1.0, y.max() + 1.0, 64)
    pdf = estimator.predict_density(X[:3], grid)
    areas = np.trapezoid(pdf, x=grid, axis=1)
    assert np.allclose(areas, 1.0, atol=0.15)
    assert np.all(pdf >= 0)
    cdf = estimator.predict_cdf(X[:3], grid)
    assert np.allclose(cdf[:, 0], 0.0, atol=1e-2)
    assert np.allclose(cdf[:, -1], 1.0, atol=1e-3)
    assert np.all(np.diff(cdf, axis=1) >= -CDF_MONOTONIC_TOL)
    samples = estimator.sample(X[:1], 5, y_grid=grid, seed=5)
    assert samples.shape == (1, 5)
    assert np.all(samples <= grid.max() + 1e-6)


def test_multi_bandwidth_heads_support_head_selection() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(72, 2))
    y = 0.5 * np.sin(X[:, 0]) + 0.25 * X[:, 1] + 0.1 * rng.normal(size=72)
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(24, 24),
        m_aux=10,
        epochs=5,
        patience=2,
        batch_size=24,
        lr=5e-3,
        bandwidths=(0.05, 0.12),
        bandwidth_strategy="mean",
        sampler="stratified",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=9).fit(X, y)
    grid = np.linspace(y.min() - 1.0, y.max() + 1.0, 48)
    pdf_mean = estimator.predict_density(X[:2], grid)
    pdf_head0 = estimator.predict_density(X[:2], grid, head=0)
    pdf_head1 = estimator.predict_density(X[:2], grid, head=1)
    assert pdf_mean.shape == pdf_head0.shape == pdf_head1.shape == (2, grid.size)
    assert not np.allclose(pdf_head0, pdf_head1)
    blend_mass = np.trapezoid(pdf_mean, x=grid, axis=1)
    head0_mass = np.trapezoid(pdf_head0, x=grid, axis=1)
    head1_mass = np.trapezoid(pdf_head1, x=grid, axis=1)
    assert np.allclose(blend_mass, 1.0, atol=0.2)
    assert np.allclose(head0_mass, 1.0, atol=0.2)
    assert np.allclose(head1_mass, 1.0, atol=0.2)
    assert np.all(pdf_mean >= 0.0)
    mean_manual = 0.5 * (pdf_head0 + pdf_head1)
    assert np.allclose(pdf_mean, mean_manual, atol=1e-5)


def test_metrics_return_finite_values() -> None:
    y_grid = np.linspace(-1, 1, 32)
    base_pdf = np.exp(-0.5 * y_grid**2)
    base_pdf /= np.trapezoid(base_pdf, x=y_grid)
    pdf = np.vstack([base_pdf, base_pdf])
    cdf = np.concatenate(
        [np.zeros((2, 1)), np.cumsum(0.5 * (pdf[:, 1:] + pdf[:, :-1]) * np.diff(y_grid), axis=1)],
        axis=1,
    )
    cdf[:, -1] = 1.0
    y_true = np.array([0.1, -0.2])
    assert nll_from_pdf(y_true, y_grid, pdf) > 0
    assert crps_from_cdf(y_true, y_grid, cdf) >= 0


def test_best_head_selection_uses_validation_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(8, 8),
        bandwidths=(0.05, 0.2),
        monitor_metric="val_nll",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=0)
    val_X = np.zeros((2, 2))
    val_y = np.array([0.1, -0.05])
    y_grid = np.linspace(-1, 1, 16)

    pdf_bad = np.full((val_X.shape[0], y_grid.size), 1e-3)
    pdf_good = np.full((val_X.shape[0], y_grid.size), 0.5)

    def fake_predict_density(
        self: CondensiteTorchCDE,
        X: np.ndarray,
        grid: np.ndarray,
        *,
        head: int | str | None = None,
    ) -> np.ndarray:
        if head == 0:
            return pdf_bad
        if head == 1:
            return pdf_good
        raise AssertionError("Unexpected head selection")

    monkeypatch.setattr(
        estimator,
        "_predict_density_internal",
        MethodType(fake_predict_density, estimator),
    )

    estimator._determine_best_head((val_X, val_y), y_grid)
    assert estimator._best_head_index == 1

    density_stack = np.stack([pdf_bad, pdf_good], axis=-1)
    combined = estimator._combine_heads(density_stack, head="best")
    assert np.allclose(combined, pdf_good)


def test_adaptive_bandwidth_option_produces_positive_values() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(96, 3))
    noise = (0.1 + 0.2 * np.abs(X[:, 0])) * rng.normal(size=X.shape[0])
    y = 0.2 * X[:, 0] - 0.4 * X[:, 1] + noise
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(16, 16),
        m_aux=12,
        epochs=4,
        patience=2,
        batch_size=24,
        lr=3e-3,
        bandwidths=(0.05, 0.12),
        adaptive_bandwidth="x",
        sampler="sobol",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=5).fit(X, y)
    assert estimator.bandwidth_net is not None
    assert estimator.x_scaler is not None
    batch = estimator.x_scaler.transform(X[:8]).astype(np.float32)
    x_tensor = torch.from_numpy(batch).to(estimator._device)
    positives = estimator._predict_adaptive_bandwidths(x_tensor)
    assert positives is not None
    assert torch.all(positives > 0)


def test_normalization_regularizer_penalty_tracks_mass_error() -> None:
    accumulator = estimator_module._NormalizationAccumulator(
        batch_size=3,
        num_heads=1,
        device=torch.device("cpu"),
    )
    predictions = torch.tensor(
        [
            [[1.0], [1.0]],
            [[0.5], [0.5]],
            [[1.5], [1.5]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    weights = torch.ones((3, 2, 1), dtype=torch.float32)

    accumulator.accumulate(predictions, weights)
    penalty = accumulator.penalty()

    assert float(penalty.detach()) == pytest.approx(1.0 / 6.0)
    penalty.backward()
    assert predictions.grad is not None
    gradients = predictions.grad.detach().cpu().numpy()
    assert np.allclose(gradients[0], 0.0)
    assert np.all(gradients[1] < 0.0)
    assert np.all(gradients[2] > 0.0)


def test_importance_sampling_strategy_trains_successfully() -> None:
    rng = np.random.default_rng(23)
    X = rng.normal(size=(90, 2))
    y = 0.3 * X[:, 0] + 0.6 * np.sin(X[:, 1]) + 0.1 * rng.normal(size=X.shape[0])
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(16, 16),
        m_aux=6,
        epochs=3,
        patience=2,
        batch_size=30,
        lr=4e-3,
        bandwidth=0.09,
        sampler="importance",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=5).fit(X, y)
    grid = np.linspace(y.min() - 0.5, y.max() + 0.5, 48)
    pdf = estimator.predict_density(X[:3], grid)
    mass = np.trapezoid(pdf, x=grid, axis=1)
    assert np.allclose(mass, 1.0, atol=0.25)


def test_quantiles_are_monotone_and_deterministic(trained_estimator) -> None:
    estimator, X, _y, grid = trained_estimator
    probs = [1e-4, 0.5, 0.999]
    quantiles = estimator.predict_quantile(X[:5], probs, y_grid=grid)
    repeat = estimator.predict_quantile(X[:5], probs, y_grid=grid)
    assert quantiles.shape == (5, 3)
    assert np.all(np.diff(quantiles, axis=1) >= -MONOTONIC_TOL)
    assert np.allclose(quantiles, repeat)


def test_predict_interval_contains_median(trained_estimator) -> None:
    estimator, X, _y, grid = trained_estimator
    lo, hi = estimator.predict_interval(X[:6], coverage=0.8, y_grid=grid)
    median = estimator.predict_quantile(X[:6], 0.5, y_grid=grid)
    assert lo.shape == hi.shape == (6,)
    assert lo.ndim == 1 and hi.ndim == 1
    assert np.all(lo <= median + 1e-8)
    assert np.all(hi >= median - 1e-8)
    lo2, hi2 = estimator.predict_interval(X[:6], coverage=0.8, y_grid=grid)
    assert np.allclose(lo, lo2)
    assert np.allclose(hi, hi2)


def test_right_tail_probability_decreases_with_threshold(trained_estimator) -> None:
    estimator, X, y, grid = trained_estimator
    low_threshold = float(y.min() - 0.5)
    mid_threshold = float(np.median(y))
    high_threshold = float(y.max() + 0.5)
    tail_low = estimator.predict_tail_prob(X[:4], threshold=low_threshold, y_grid=grid)
    tail_mid = estimator.predict_tail_prob(X[:4], threshold=mid_threshold, y_grid=grid)
    tail_high = estimator.predict_tail_prob(X[:4], threshold=high_threshold, y_grid=grid)
    assert tail_low.shape == tail_mid.shape == tail_high.shape == (4,)
    assert np.all(tail_high <= tail_mid)
    assert np.all(tail_mid <= tail_low)
    high_prob_floor = 0.9
    high_prob_ceil = 0.1
    assert np.all(tail_low >= high_prob_floor)
    assert np.all(tail_high <= high_prob_ceil)


def test_left_expected_shortfall_is_below_lower_tail_quantile(trained_estimator) -> None:
    estimator, X, _y, grid = trained_estimator
    alpha = 0.9
    lower_quantile = estimator.predict_quantile(X[:6], 1.0 - alpha, y_grid=grid)
    es_values = estimator.expected_shortfall(X[:6], alpha=alpha, side="left", y_grid=grid)
    assert lower_quantile.shape == es_values.shape == (6,)
    assert np.all(es_values <= lower_quantile + 1e-6)


def test_expected_shortfall_exceeds_quantile(trained_estimator) -> None:
    estimator, X, _y, grid = trained_estimator
    alpha = 0.9
    quantile = estimator.predict_quantile(X[:6], alpha, y_grid=grid)
    es_values = estimator.expected_shortfall(X[:6], alpha=alpha, y_grid=grid)
    assert quantile.shape == es_values.shape == (6,)
    assert np.all(es_values >= quantile - 1e-6)
    repeat = estimator.expected_shortfall(X[:6], alpha=alpha, y_grid=grid)
    assert np.allclose(es_values, repeat)
