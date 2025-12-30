"""Report how streamed auxiliary chunks reduce expanded feature rows."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace

import numpy as np

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig


def _make_dataset(n_samples: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n_samples, 4))
    noise = 0.2 * rng.normal(size=n_samples)
    y = 0.3 * X[:, 0] - 0.4 * X[:, 1] + noise
    return X, y


@contextmanager
def _track_feature_rows(stats: dict[str, int]) -> None:
    original = CondensiteTorchCDE._combine_features

    def spy(self: CondensiteTorchCDE, X_batch, y_chunk):
        features = original(self, X_batch, y_chunk)
        stats["max_rows"] = max(stats["max_rows"], features.shape[0])
        return features

    setattr(CondensiteTorchCDE, "_combine_features", spy)
    try:
        yield
    finally:
        setattr(CondensiteTorchCDE, "_combine_features", original)


def _run_with_config(config: CondensiteTorchCDEConfig, X: np.ndarray, y: np.ndarray) -> dict[str, int]:
    stats = {"max_rows": 0}
    with _track_feature_rows(stats):
        CondensiteTorchCDE(config=config, random_seed=0).fit(X, y)
    chunk = config.aux_chunk_size if config.aux_chunk_size is not None else min(config.m_aux, 64)
    return {
        "max_feature_rows": stats["max_rows"],
        "aux_chunk_size": chunk if chunk > 0 else config.m_aux,
    }


def main() -> None:
    X, y = _make_dataset()
    base = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=256,
        batch_size=128,
        epochs=2,
        patience=1,
        sampler="sobol",
    )
    configs = {
        "full_expansion": replace(base, aux_chunk_size=0),
        "streamed_chunks": replace(base, aux_chunk_size=16),
    }
    results: dict[str, dict[str, int]] = {}
    for label, cfg in configs.items():
        results[label] = _run_with_config(cfg, X, y)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
