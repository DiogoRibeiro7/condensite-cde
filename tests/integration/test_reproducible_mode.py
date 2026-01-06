"""Integration tests covering reproducible mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

pytestmark = pytest.mark.integration


def _toy_data(n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 2))
    y = 0.4 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_reproducible_mode_matches_predictions() -> None:
    X, y = _toy_data()
    config = CondensiteTorchCDEConfig(
        epochs=4,
        patience=2,
        m_aux=16,
        batch_size=32,
        val_fraction=0.2,
        reproducible=True,
    )
    estimator_one = CondensiteTorchCDE(config=config, random_seed=11).fit(X, y)
    estimator_two = CondensiteTorchCDE(config=config, random_seed=11).fit(X, y)
    grid = np.linspace(y.min() - 0.5, y.max() + 0.5, 64)
    pdf_one = estimator_one.predict_density(X[:8], grid)
    pdf_two = estimator_two.predict_density(X[:8], grid)
    assert np.allclose(pdf_one, pdf_two)


def test_model_card_persisted(tmp_path: Path) -> None:
    X, y = _toy_data()
    config = CondensiteTorchCDEConfig(
        epochs=3,
        patience=1,
        m_aux=12,
        val_fraction=0.0,
        reproducible=True,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=5).fit(X, y)
    card = estimator.model_card()
    assert "config_hash" in card
    assert card["training"]["epochs_requested"] == config.epochs
    target_dir = tmp_path / "artifacts"
    estimator.save(target_dir)
    restored = CondensiteTorchCDE.load(target_dir)
    assert restored.model_card()["config_hash"] == card["config_hash"]
