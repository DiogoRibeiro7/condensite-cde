from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - Torch availability
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

pytestmark = pytest.mark.integration


def test_training_streams_auxiliary_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(96, 2))
    y = 0.5 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * rng.normal(size=96)
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(16, 16),
        m_aux=48,
        aux_chunk_size=6,
        batch_size=12,
        epochs=1,
        patience=1,
        sampler="sobol",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=0)
    seen_rows: list[int] = []
    original = CondensiteTorchCDE._combine_features

    def spy(self: CondensiteTorchCDE, X_batch, y_chunk):
        features = original(self, X_batch, y_chunk)
        seen_rows.append(features.shape[0])
        return features

    monkeypatch.setattr(CondensiteTorchCDE, "_combine_features", spy)
    estimator.fit(X, y)
    assert seen_rows, "Expected at least one streamed chunk."
    chunk = estimator._effective_aux_chunk_size()
    assert chunk <= config.aux_chunk_size
    assert max(seen_rows) <= config.batch_size * chunk
