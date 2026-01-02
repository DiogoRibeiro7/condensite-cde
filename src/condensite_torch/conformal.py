"""Split-conformal intervals constructed from a Condensite estimator."""

from __future__ import annotations

import copy
import math
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from .diagnostics import pit_values
from .estimator import CondensiteTorchCDE, CondensiteTorchCDEConfig


class ConformalCDEWrapper:
    """Wrap an estimator with split-conformal predictive intervals."""

    def __init__(
        self,
        base_config: CondensiteTorchCDEConfig,
        *,
        random_seed: int = 0,
    ) -> None:
        self.base_config = copy.deepcopy(base_config)
        self.random_seed = int(random_seed)
        self.estimator = CondensiteTorchCDE(
            config=copy.deepcopy(base_config),
            random_seed=random_seed,
        )
        self._X_cal: NDArray[np.float64] | None = None
        self._y_cal: NDArray[np.float64] | None = None
        self._fitted = False
        self._method: Literal["quantile", "cdf"] = "quantile"

    def fit(
        self,
        X_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        X_cal: NDArray[np.floating],
        y_cal: NDArray[np.floating],
        *,
        method: Literal["quantile", "cdf"] = "quantile",
    ) -> ConformalCDEWrapper:
        X_train_arr = np.asarray(X_train, dtype=np.float64)
        y_train_arr = np.asarray(y_train, dtype=np.float64).reshape(-1)
        X_cal_arr = np.asarray(X_cal, dtype=np.float64)
        y_cal_arr = np.asarray(y_cal, dtype=np.float64).reshape(-1)
        if X_cal_arr.shape[0] != y_cal_arr.shape[0]:
            msg = "Calibration X and y must contain the same number of samples."
            raise ValueError(msg)
        method_normalized = method.lower()
        if method_normalized not in {"quantile", "cdf"}:
            msg = f"method must be 'quantile' or 'cdf', got {method}"
            raise ValueError(msg)
        self.estimator.fit(X_train_arr, y_train_arr)
        self._X_cal = X_cal_arr
        self._y_cal = y_cal_arr
        self._fitted = True
        self._method = cast(Literal["quantile", "cdf"], method_normalized)
        return self

    def predict_interval(
        self,
        X: NDArray[np.floating],
        *,
        coverage: float = 0.9,
        y_grid: NDArray[np.floating] | None = None,
        head: int | str | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return split-conformal prediction intervals with the requested coverage."""
        self._ensure_fitted()
        if not 0.0 < coverage < 1.0:
            msg = f"coverage must lie in (0, 1), got {coverage}"
            raise ValueError(msg)
        alpha = 1.0 - float(coverage)
        grid = self._resolve_y_grid(y_grid)
        if self._method == "quantile":
            tail = alpha / 2.0
            cal_slack = self._quantile_slack(
                coverage=coverage,
                tail=tail,
                y_grid=grid,
                head=head,
            )
            quantiles = self.estimator.predict_quantile(
                X,
                [tail, 1.0 - tail],
                y_grid=grid,
                head=head,
            )
            lower = quantiles[:, 0] - cal_slack
            upper = quantiles[:, 1] + cal_slack
        else:
            tail_prob = self._cdf_tail_probability(
                alpha=alpha,
                y_grid=grid,
                head=head,
            )
            probs = np.array([tail_prob, 1.0 - tail_prob], dtype=np.float64)
            quantiles = self.estimator.predict_quantile(
                X,
                probs,
                y_grid=grid,
                head=head,
            )
            lower, upper = quantiles[:, 0], quantiles[:, 1]
        return lower.astype(np.float64, copy=False), upper.astype(np.float64, copy=False)

    def _quantile_slack(
        self,
        *,
        coverage: float,
        tail: float,
        y_grid: NDArray[np.float64],
        head: int | str | None,
    ) -> float:
        assert self._X_cal is not None
        assert self._y_cal is not None
        quantiles = self.estimator.predict_quantile(
            self._X_cal,
            [tail, 1.0 - tail],
            y_grid=y_grid,
            head=head,
        )
        lower = quantiles[:, 0]
        upper = quantiles[:, 1]
        residual_low = lower - self._y_cal
        residual_high = self._y_cal - upper
        scores = np.maximum.reduce([np.zeros_like(residual_low), residual_low, residual_high])
        sorted_scores = np.sort(scores)
        n = sorted_scores.size
        rank = min(n - 1, max(0, int(math.ceil((n + 1) * coverage)) - 1))
        return float(sorted_scores[rank])

    def _cdf_tail_probability(
        self,
        *,
        alpha: float,
        y_grid: NDArray[np.float64],
        head: int | str | None,
    ) -> float:
        assert self._X_cal is not None
        assert self._y_cal is not None
        cdf = self.estimator.predict_cdf(
            self._X_cal,
            y_grid,
            head=head,
        )
        pit = pit_values(self._y_cal, y_grid, cdf)
        scores = np.minimum(pit, 1.0 - pit)
        sorted_scores = np.sort(scores)
        n = sorted_scores.size
        rank = min(n - 1, max(0, int(math.ceil((n + 1) * alpha)) - 1))
        tail = float(sorted_scores[rank])
        return float(np.clip(tail, 1e-6, 0.5))

    def _resolve_y_grid(self, provided: NDArray[np.floating] | None) -> NDArray[np.float64]:
        if provided is not None:
            return self.estimator._validate_y_grid(provided)  # noqa: SLF001
        return self.estimator._default_y_grid()  # noqa: SLF001

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            msg = "Call fit() with train/calibration splits before requesting intervals."
            raise RuntimeError(msg)


__all__ = ("ConformalCDEWrapper",)
