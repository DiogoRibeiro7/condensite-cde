from __future__ import annotations

import numpy as np
import pytest

from condensite_cde import cross_validate
from condensite_cde import cv as cv_module
from condensite_torch import CondensiteTorchCDEConfig


def _toy_dataset(n_samples: int = 72) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n_samples, 3))
    noise = 0.05 * rng.normal(size=n_samples)
    y = 0.4 * X[:, 0] - 0.2 * X[:, 1] + noise
    return X, y


def test_cross_validate_reports_means_and_stds() -> None:
    X, y = _toy_dataset()
    config = CondensiteTorchCDEConfig(
        epochs=3,
        patience=2,
        batch_size=32,
        m_aux=16,
        sampler="sobol",
    )
    result = cross_validate(
        config,
        X,
        y,
        cv=3,
        metrics=("nll", "crps", "coverage"),
        seed=2,
    )
    assert set(result.metrics_mean) == {"nll", "crps", "coverage"}
    assert len(result.folds) == result.metadata["cv"]
    manual_mean = float(np.mean([fold.metrics["nll"] for fold in result.folds]))
    assert pytest.approx(result.metrics_mean["nll"], rel=1e-12) == manual_mean
    manual_std = float(np.std([fold.metrics["nll"] for fold in result.folds], ddof=0))
    assert pytest.approx(result.metrics_std["nll"], rel=1e-12) == manual_std


def test_cross_validate_is_deterministic_with_seed() -> None:
    X, y = _toy_dataset(60)
    config = CondensiteTorchCDEConfig(
        epochs=2,
        patience=1,
        batch_size=24,
        m_aux=12,
        sampler="sobol",
    )
    result_a = cross_validate(config, X, y, cv=2, metrics=("nll",), seed=99)
    result_b = cross_validate(config, X, y, cv=2, metrics=("nll",), seed=99)
    assert result_a.to_dict() == result_b.to_dict()


def test_group_folds_keep_groups_together() -> None:
    groups = np.repeat(np.arange(6), 3)
    rng = np.random.default_rng(4)
    folds = cv_module._make_group_folds(groups, cv=3, rng=rng)
    assigned = np.concatenate(folds)
    assert sorted(assigned.tolist()) == list(range(groups.shape[0]))
    for fold in folds:
        fold_groups = np.unique(groups[fold])
        for group in fold_groups:
            assert np.all(groups[fold][groups[fold] == group] == group)
