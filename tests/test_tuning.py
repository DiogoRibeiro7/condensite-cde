"""Tests for tuning utilities and validation-driven early stopping."""

from __future__ import annotations

import importlib

import numpy as np

from condensite_cde.tune import tune_bandwidth_m_aux


def _make_small_dataset(n_samples: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 2))
    noise = (0.2 + 0.1 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.5 * X[:, 0] - 0.25 * X[:, 1] + noise
    return X, y


def test_training_history_contains_validation_metrics(torch_available: bool) -> None:
    assert torch_available
    module = importlib.import_module("condensite_torch")
    CondensiteTorchCDE = module.CondensiteTorchCDE
    CondensiteTorchCDEConfig = module.CondensiteTorchCDEConfig
    X, y = _make_small_dataset()
    config = CondensiteTorchCDEConfig(
        epochs=4,
        patience=1,
        batch_size=32,
        m_aux=16,
        bandwidth=0.15,
        sampler="stratified",
        val_fraction=0.2,
        monitor_metric="val_nll",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=2).fit(X, y)
    assert estimator.training_history, "Expected non-empty history"
    assert len(estimator.training_history) <= config.epochs
    for record in estimator.training_history:
        assert "val_crps" in record
        assert "val_nll" in record


def test_tuner_returns_best_configuration(torch_available: bool) -> None:
    assert torch_available
    module = importlib.import_module("condensite_torch")
    CondensiteTorchCDEConfig = module.CondensiteTorchCDEConfig
    X, y = _make_small_dataset()
    base_config = CondensiteTorchCDEConfig(
        epochs=3,
        patience=1,
        batch_size=32,
        val_fraction=0.2,
        sampler="iid",
    )
    bandwidths = [0.08, 0.12]
    aux_values = [12, 16]
    result = tune_bandwidth_m_aux(
        X,
        y,
        bandwidths=bandwidths,
        m_aux_values=aux_values,
        base_config=base_config,
        metric="val_crps",
        random_seed=10,
    )
    assert result.best_config.bandwidth in {0.08, 0.12}
    assert result.best_config.m_aux in {12, 16}
    assert len(result.history) == len(bandwidths) * len(aux_values)
    assert result.metric_name == "val_crps"
