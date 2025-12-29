"""Persistence tests for CondensiteTorchCDE."""

from __future__ import annotations

import numpy as np


def test_save_and_load_round_trip(tmp_path, trained_estimator, torch_available) -> None:
    from condensite_torch import CondensiteTorchCDE  # noqa: PLC0415

    estimator, X, _, grid = trained_estimator
    save_dir = tmp_path / "artifact"
    estimator.save(save_dir)
    restored = CondensiteTorchCDE.load(save_dir, map_location="cpu")
    original_pdf = estimator.predict_density(X[:2], grid)
    loaded_pdf = restored.predict_density(X[:2], grid)
    assert np.allclose(original_pdf, loaded_pdf, atol=1e-5)
