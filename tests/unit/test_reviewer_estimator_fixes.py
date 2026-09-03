"""Regression tests for estimator findings raised during PR #12 review."""

from __future__ import annotations

import numpy as np
import pytest
from torch import nn

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

pytestmark = pytest.mark.unit


def test_ensemble_epistemic_mode_fails_explicitly() -> None:
    config = CondensiteTorchCDEConfig(epistemic_mode="ensemble")
    with pytest.raises(ValueError, match="EnsembleCondensite"):
        CondensiteTorchCDE(config=config)


def test_empty_inference_batch_uses_positive_loop_step() -> None:
    estimator = CondensiteTorchCDE()
    minimum_batch_size = 1
    assert estimator._inference_row_batch_size(0) == minimum_batch_size


def test_derived_schema_does_not_freeze_training_target_range() -> None:
    estimator = CondensiteTorchCDE()
    X = np.zeros((3, 2), dtype=np.float64)
    y = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    schema = estimator._schema_from_training(X, y)
    assert schema is not None
    assert schema.y_min is None
    assert schema.y_max is None


def test_mc_dropout_enables_dropout_and_restores_eval_mode() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(32, 2))
    y = 0.4 * X[:, 0] - 0.2 * X[:, 1] + 0.05 * rng.normal(size=32)
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(8,),
        dropout=0.5,
        epistemic_mode="mc_dropout",
        mc_samples=3,
        epochs=1,
        patience=1,
        batch_size=16,
        m_aux=8,
        val_fraction=0.2,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=7).fit(X, y)
    assert estimator.model is not None

    dropout_modules = [
        module for module in estimator.model.modules() if isinstance(module, nn.Dropout)
    ]
    assert dropout_modules
    observed_modes: list[bool] = []
    handles = [
        module.register_forward_pre_hook(
            lambda current_module, _inputs: observed_modes.append(current_module.training),
        )
        for module in dropout_modules
    ]

    estimator.model.eval()
    grid = np.linspace(float(y.min()), float(y.max()), 12)
    try:
        output = estimator._predict_density_mc_dropout(X[:2], grid)
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == (2, grid.size)
    assert observed_modes
    assert all(observed_modes)
    assert estimator.model.training is False
