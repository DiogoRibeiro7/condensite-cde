"""Multi-target wrapper supporting independent, autoregressive, and shared-trunk outputs."""

from __future__ import annotations

import copy
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from .estimator import CondensiteTorchCDE, CondensiteTorchCDEConfig

_EXPECTED_CONTEXT_DIM = 2
_HISTORY_STACK_DIMENSION = 3

MultiTargetMode = Literal["independent", "autoregressive", "shared"]


class MultiTargetCondensite:
    """Fit one estimator per target dimension with optional autoregressive conditioning."""

    def __init__(
        self,
        base_config: CondensiteTorchCDEConfig,
        *,
        mode: MultiTargetMode = "independent",
        random_seed: int = 0,
    ) -> None:
        mode_lower = mode.lower()
        if mode_lower not in {"independent", "autoregressive", "shared"}:
            msg = "mode must be 'independent', 'autoregressive', or 'shared'"
            raise ValueError(msg)
        self.base_config = base_config
        self.mode = cast(MultiTargetMode, mode_lower)
        self.random_seed = int(random_seed)
        self._models: list[CondensiteTorchCDE] = []
        self._shared_estimator: CondensiteTorchCDE | None = None
        self._dimension = 0
        self._fitted = False

    def fit(self, X: NDArray[np.floating], Y: NDArray[np.floating]) -> MultiTargetCondensite:
        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.float64)
        if X_arr.ndim != _EXPECTED_CONTEXT_DIM:
            msg = f"X must be 2-D, got {X_arr.shape}"
            raise ValueError(msg)
        if Y_arr.ndim != _EXPECTED_CONTEXT_DIM:
            msg = f"Y must be 2-D, got {Y_arr.shape}"
            raise ValueError(msg)
        if X_arr.shape[0] != Y_arr.shape[0]:
            msg = "X and Y must share the first dimension."
            raise ValueError(msg)
        self._models.clear()
        self._shared_estimator = None
        self._dimension = Y_arr.shape[1]
        if self.mode == "shared":
            features, targets = self._flatten_shared_dataset(X_arr, Y_arr)
            config = copy.deepcopy(self.base_config)
            estimator = CondensiteTorchCDE(config=config, random_seed=self.random_seed)
            estimator.fit(features, targets)
            self._shared_estimator = estimator
            self._fitted = True
            return self
        for dim in range(self._dimension):
            if self.mode == "independent":
                features = X_arr
            else:
                features = self._augment_features(X_arr, Y_arr[:, :dim])
            config = copy.deepcopy(self.base_config)
            estimator = CondensiteTorchCDE(config=config, random_seed=self.random_seed + dim)
            estimator.fit(features, Y_arr[:, dim])
            self._models.append(estimator)
        self._fitted = True
        return self

    def predict_density(
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating],
        *,
        y_context: NDArray[np.floating] | None = None,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        grid = np.asarray(y_grid, dtype=np.float64)
        if self.mode == "shared":
            if y_context is not None:
                msg = "y_context is not supported in shared mode."
                raise ValueError(msg)
            estimator = self._require_shared_estimator()
            shared_features = self._tile_with_target_indicator(X_arr)
            density = estimator.predict_density(shared_features, grid, head=head)
            return density.reshape(X_arr.shape[0], self._dimension, grid.size)
        per_dim = []
        for dim, estimator in enumerate(self._models):
            features = self._features_for_prediction(X_arr, y_context, dim)
            density = estimator.predict_density(features, grid, head=head)
            per_dim.append(density[:, None, :])
        return np.concatenate(per_dim, axis=1)

    def predict_cdf(
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating],
        *,
        y_context: NDArray[np.floating] | None = None,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        density = self.predict_density(X, y_grid, y_context=y_context, head=head)
        grid = np.asarray(y_grid, dtype=np.float64)
        cdfs = np.empty_like(density)
        if self.mode == "shared":
            estimator = self._require_shared_estimator()
            flat_density = density.reshape(-1, grid.size)
            cdf = estimator._cdf_from_pdf(flat_density, grid)
            cdfs[:] = cdf.reshape(X.shape[0], self._dimension, grid.size)
            return cdfs
        for dim, estimator in enumerate(self._models):
            per_dim_density = density[:, dim, :]
            cdf = estimator._cdf_from_pdf(per_dim_density, grid)
            cdfs[:, dim, :] = cdf
        return cdfs

    def predict_quantile(
        self,
        X: NDArray[np.floating],
        q: NDArray[np.floating] | float,
        *,
        y_context: NDArray[np.floating] | None = None,
        y_grid: NDArray[np.floating] | None = None,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        scalar = q_arr.size == 1 and np.ndim(q) == 0
        if self.mode == "shared":
            estimator = self._require_shared_estimator()
            features = self._tile_with_target_indicator(X_arr)
            values = estimator.predict_quantile(features, q_arr, y_grid=y_grid, head=head)
            stacked = values.reshape(X_arr.shape[0], self._dimension, q_arr.size)
            if scalar:
                return stacked[..., 0]
            return stacked
        per_dim = []
        for dim, estimator in enumerate(self._models):
            features = self._features_for_prediction(X_arr, y_context, dim)
            values = estimator.predict_quantile(features, q_arr, y_grid=y_grid, head=head)
            per_dim.append(values[:, None, :])
        stacked = np.concatenate(per_dim, axis=1)
        if scalar:
            return stacked[..., 0]
        return stacked

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
        X_arr = np.asarray(X, dtype=np.float64)
        n_obs = X_arr.shape[0]
        rng = np.random.default_rng(self.random_seed if seed is None else seed)
        samples = np.zeros((n_obs, n_samples, self._dimension), dtype=np.float64)
        if self.mode == "shared":
            estimator = self._require_shared_estimator()
            features = self._tile_with_target_indicator(X_arr)
            draws = estimator.sample(features, n_samples, seed=rng.integers(0, 2**32 - 1))
            samples[:] = draws.reshape(n_obs, self._dimension, n_samples).transpose(0, 2, 1)
            return samples
        for dim, estimator in enumerate(self._models):
            draw_seed = int(rng.integers(0, 2**32 - 1))
            if self.mode == "independent":
                draws = estimator.sample(X_arr, n_samples, seed=draw_seed)
            else:
                history = samples[:, :, :dim]
                features = self._repeat_with_history(X_arr, history)
                draws = estimator.sample(features, 1, seed=draw_seed).reshape(n_obs, n_samples)
            samples[:, :, dim] = draws
        return samples

    def _features_for_prediction(
        self,
        X: NDArray[np.float64],
        y_context: NDArray[np.floating] | None,
        dim: int,
    ) -> NDArray[np.float64]:
        if self.mode == "independent":
            return X
        if self.mode == "shared":
            if y_context is not None:
                msg = "y_context is not supported in shared mode."
                raise ValueError(msg)
            return self._tile_with_target_indicator(X)
        if dim == 0:
            return X
        if y_context is None:
            msg = "y_context must be provided for autoregressive predictions."
            raise ValueError(msg)
        context = np.asarray(y_context, dtype=np.float64)
        if context.ndim != _EXPECTED_CONTEXT_DIM or context.shape[1] < dim:
            msg = f"y_context must have shape (n, >= {dim})"
            raise ValueError(msg)
        return self._augment_features(X, context[:, :dim])

    @staticmethod
    def _augment_features(
        X: NDArray[np.float64],
        prefix: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        if prefix is None or prefix.size == 0:
            return X
        prefix_arr = np.asarray(prefix, dtype=np.float64)
        if prefix_arr.ndim == 1:
            prefix_arr = prefix_arr.reshape(-1, 1)
        combined = np.concatenate([X, prefix_arr.astype(np.float64, copy=False)], axis=1)
        return combined.astype(np.float64, copy=False)

    def _repeat_with_history(
        self,
        X: NDArray[np.float64],
        history: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n_obs = X.shape[0]
        if history.size == 0:
            repeats = history.shape[1] if history.ndim == _HISTORY_STACK_DIMENSION else 1
            return np.repeat(X, repeats, axis=0)
        n_samples = history.shape[1]
        history_flat = history.reshape(n_obs * n_samples, -1)
        X_rep = np.repeat(X, n_samples, axis=0)
        return self._augment_features(X_rep, history_flat)

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            msg = "Call fit() before requesting predictions."
            raise RuntimeError(msg)

    def _flatten_shared_dataset(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        indicator = np.tile(np.eye(self._dimension, dtype=np.float64), (X.shape[0], 1))
        repeated_X = np.repeat(X, self._dimension, axis=0)
        features = np.concatenate([repeated_X, indicator], axis=1)
        targets = Y.reshape(-1)
        return features, targets

    def _tile_with_target_indicator(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        indicator = np.tile(np.eye(self._dimension, dtype=np.float64), (X.shape[0], 1))
        repeated_X = np.repeat(X, self._dimension, axis=0)
        return np.concatenate([repeated_X, indicator], axis=1)

    def _require_shared_estimator(self) -> CondensiteTorchCDE:
        if self._shared_estimator is None:
            msg = "Shared estimator not initialized; call fit() first."
            raise RuntimeError(msg)
        return self._shared_estimator


__all__ = ("MultiTargetCondensite",)
