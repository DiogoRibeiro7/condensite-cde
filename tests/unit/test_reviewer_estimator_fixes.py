"""Regression tests for estimator findings raised during PR #12 review."""

from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
from torch import nn

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig
from condensite_torch.scalers import MinMaxScaler1D, StandardScaler

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


def test_mc_dropout_preserves_training_mode_for_internal_passes() -> None:
    samples = 3
    rows = 2
    grid_points = 8
    config = CondensiteTorchCDEConfig(epistemic_mode="mc_dropout", mc_samples=samples)
    estimator = CondensiteTorchCDE(config=config)
    estimator._fitted = True
    estimator.model = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(1, 1))
    estimator.x_scaler = StandardScaler().fit(np.zeros((rows, 1), dtype=np.float64))
    estimator.y_scaler = MinMaxScaler1D().fit(np.array([-1.0, 1.0], dtype=np.float64))
    calls: list[tuple[bool, bool]] = []

    def fake_predict_density_internal(
        self: CondensiteTorchCDE,
        X: np.ndarray,
        y_grid: np.ndarray,
        *,
        head: int | str | None = None,
        force_eval: bool = True,
    ) -> np.ndarray:
        del head
        calls.append((force_eval, self.model.training if self.model is not None else False))
        return np.ones((X.shape[0], y_grid.size), dtype=np.float64)

    estimator._predict_density_internal = MethodType(fake_predict_density_internal, estimator)
    X = np.zeros((rows, 1), dtype=np.float64)
    grid = np.linspace(-1.0, 1.0, grid_points)
    output = estimator._predict_density_mc_dropout(X, grid)

    assert output.shape == (rows, grid_points)
    assert calls == [(False, True)] * samples
    assert estimator.model.training is True
