"""Autoregressive wrapper around Condensite Torch for multivariate targets."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .estimator import CondensiteTorchCDE, CondensiteTorchCDEConfig


@dataclass
class AutoregressiveCondensite:
    base_config: CondensiteTorchCDEConfig
    random_seed: int = 0

    def __post_init__(self) -> None:
        self._models: list[CondensiteTorchCDE] = []
        self._dimension: int = 0
        self._fitted = False

    def fit(self, X: NDArray[np.floating], Y: NDArray[np.floating]) -> "AutoregressiveCondensite":
        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.float64)
        if X_arr.ndim != 2:
            msg = f"X must be 2-D, got {X_arr.shape}"
            raise ValueError(msg)
        if Y_arr.ndim != 2:
            msg = f"Y must be 2-D, got {Y_arr.shape}"
            raise ValueError(msg)
        if X_arr.shape[0] != Y_arr.shape[0]:
            msg = "X and Y must share the first dimension."
            raise ValueError(msg)
        n_targets = Y_arr.shape[1]
        self._dimension = n_targets
        self._models.clear()
        for dim in range(n_targets):
            features = self._augment_features(X_arr, Y_arr[:, :dim])
            config = copy.deepcopy(self.base_config)
            estimator = CondensiteTorchCDE(config=config, random_seed=self.random_seed + dim)
            estimator.fit(features, Y_arr[:, dim])
            self._models.append(estimator)
        self._fitted = True
        return self

    def sample(
        self,
        X: NDArray[np.floating],
        n_samples: int,
        *,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        self._ensure_fitted()
        if n_samples <= 0:
            msg = "n_samples must be positive."
            raise ValueError(msg)
        rng = np.random.default_rng(self.random_seed if seed is None else seed)
        X_arr = np.asarray(X, dtype=np.float64)
        n_obs = X_arr.shape[0]
        samples = np.zeros((n_obs, n_samples, self._dimension), dtype=np.float64)
        for dim, estimator in enumerate(self._models):
            history = samples[:, :, :dim]
            features = self._repeat_with_history(X_arr, history)
            draw_seed = int(rng.integers(0, 2**32 - 1))
            draws = estimator.sample(features, 1, seed=draw_seed).reshape(n_obs, n_samples)
            samples[:, :, dim] = draws
        return samples

    def predict_marginal_quantile(
        self,
        X: NDArray[np.floating],
        dim: int,
        q: NDArray[np.floating] | float,
        *,
        y_prefix: NDArray[np.floating] | None = None,
        y_grid: NDArray[np.floating] | None = None,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        self._ensure_fitted()
        if dim < 0 or dim >= self._dimension:
            msg = f"dim must be in [0, {self._dimension}), got {dim}"
            raise ValueError(msg)
        X_arr = np.asarray(X, dtype=np.float64)
        prefix = None
        if dim > 0:
            if y_prefix is None:
                msg = "y_prefix must be provided for dim > 0."
                raise ValueError(msg)
            prefix = np.asarray(y_prefix, dtype=np.float64)
            if prefix.shape != (X_arr.shape[0], dim):
                msg = f"y_prefix must have shape (n, {dim})"
                raise ValueError(msg)
        else:
            prefix = None
        features = self._augment_features(X_arr, prefix)
        estimator = self._models[dim]
        return estimator.predict_quantile(features, q, y_grid=y_grid, head=head)

    def _repeat_with_history(
        self,
        X: NDArray[np.float64],
        history: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n_obs = X.shape[0]
        if history.size == 0:
            return np.repeat(X, history.shape[1] if history.ndim == 3 else 1, axis=0)
        n_samples = history.shape[1]
        history_flat = history.reshape(n_obs * n_samples, -1)
        X_rep = np.repeat(X, n_samples, axis=0)
        return self._augment_features(X_rep, history_flat)

    @staticmethod
    def _augment_features(
        X: NDArray[np.float64],
        prefix: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        if prefix is None or prefix.size == 0:
            return X
        return np.concatenate([X, prefix], axis=1)

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            msg = "Call fit() before requesting samples or quantiles."
            raise RuntimeError(msg)
