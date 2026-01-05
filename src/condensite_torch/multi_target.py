"""Multi-target wrapper supporting independent and autoregressive outputs."""

from __future__ import annotations

import copy
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from .estimator import CondensiteTorchCDE, CondensiteTorchCDEConfig

MultiTargetMode = Literal["independent", "autoregressive"]


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
        if mode_lower not in {"independent", "autoregressive"}:
            msg = "mode must be 'independent' or 'autoregressive'"
            raise ValueError(msg)
        self.base_config = base_config
        self.mode = cast(MultiTargetMode, mode_lower)
        self.random_seed = int(random_seed)
        self._models: list[CondensiteTorchCDE] = []
        self._dimension = 0
        self._fitted = False

    def fit(self, X: NDArray[np.floating], Y: NDArray[np.floating]) -> MultiTargetCondensite:
        X_arr = np.asarray(X, dtype=object)
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
        self._models.clear()
        self._dimension = Y_arr.shape[1]
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
        X_arr = np.asarray(X, dtype=object)
        grid = np.asarray(y_grid, dtype=np.float64)
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
        for dim, estimator in enumerate(self._models):
            per_dim_density = density[:, dim, :]
            cdf = estimator._cdf_from_pdf(per_dim_density, grid)  # noqa: SLF001
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
        X_arr = np.asarray(X, dtype=object)
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        scalar = q_arr.size == 1 and np.ndim(q) == 0
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
        X_arr = np.asarray(X, dtype=object)
        n_obs = X_arr.shape[0]
        rng = np.random.default_rng(self.random_seed if seed is None else seed)
        samples = np.zeros((n_obs, n_samples, self._dimension), dtype=np.float64)
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
        X: NDArray[object],
        y_context: NDArray[np.floating] | None,
        dim: int,
    ) -> NDArray[object]:
        if self.mode == "independent":
            return X
        if dim == 0:
            return X
        if y_context is None:
            msg = "y_context must be provided for autoregressive predictions."
            raise ValueError(msg)
        context = np.asarray(y_context, dtype=np.float64)
        if context.ndim != 2 or context.shape[1] < dim:
            msg = f"y_context must have shape (n, >= {dim})"
            raise ValueError(msg)
        return self._augment_features(X, context[:, :dim])

    @staticmethod
    def _augment_features(
        X: NDArray[object],
        prefix: NDArray[np.float64] | None,
    ) -> NDArray[object]:
        if prefix is None or prefix.size == 0:
            return X
        prefix_arr = np.asarray(prefix, dtype=np.float64)
        if prefix_arr.ndim == 1:
            prefix_arr = prefix_arr.reshape(-1, 1)
        combined = np.concatenate([X, prefix_arr.astype(object)], axis=1)
        return combined

    def _repeat_with_history(
        self,
        X: NDArray[object],
        history: NDArray[np.float64],
    ) -> NDArray[object]:
        n_obs = X.shape[0]
        if history.size == 0:
            return np.repeat(X, history.shape[1] if history.ndim == 3 else 1, axis=0)
        n_samples = history.shape[1]
        history_flat = history.reshape(n_obs * n_samples, -1)
        X_rep = np.repeat(X, n_samples, axis=0)
        return self._augment_features(X_rep, history_flat)

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            msg = "Call fit() before requesting predictions."
            raise RuntimeError(msg)


__all__ = ("MultiTargetCondensite",)
