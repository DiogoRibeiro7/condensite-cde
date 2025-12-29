"""Deterministic NumPy-based scalers used by the estimator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

_EXPECTED_FEATURE_DIM = 2


def _ensure_2d(X: NDArray[np.floating]) -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != _EXPECTED_FEATURE_DIM:
        msg = f"Expected 2D input, got shape {arr.shape}"
        raise ValueError(msg)
    return arr


@dataclass
class StandardScaler:
    """Column-wise mean/variance scaler for tabular covariates."""

    eps: float = 1e-8
    mean_: NDArray[np.float64] = field(init=False)
    scale_: NDArray[np.float64] = field(init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, X: NDArray[np.floating]) -> StandardScaler:
        arr = _ensure_2d(X)
        self.mean_ = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale < self.eps] = 1.0
        self.scale_ = scale
        self.fitted_ = True
        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.float32]:
        self._check_is_fitted()
        arr = _ensure_2d(X)
        return ((arr - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, X: NDArray[np.floating]) -> NDArray[np.float64]:
        self._check_is_fitted()
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return (arr * self.scale_) + self.mean_

    def _check_is_fitted(self) -> None:
        if not self.fitted_:
            msg = "StandardScaler must be fitted before calling transform."
            raise RuntimeError(msg)


@dataclass
class MinMaxScaler1D:
    """Min/max scaler that maps values to [0, 1] deterministically."""

    eps: float = 1e-8
    min_: float = field(init=False)
    max_: float = field(init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, y: NDArray[np.floating]) -> MinMaxScaler1D:
        arr = np.asarray(y, dtype=np.float64).reshape(-1)
        self.min_ = float(arr.min())
        self.max_ = float(arr.max())
        if self.max_ - self.min_ < self.eps:
            # Avoid division by zero for constant targets.
            self.max_ = self.min_ + self.eps
        self.fitted_ = True
        return self

    @property
    def data_range_(self) -> float:
        self._check_is_fitted()
        return self.max_ - self.min_

    def transform(self, y: NDArray[np.floating]) -> NDArray[np.float32]:
        self._check_is_fitted()
        arr = np.asarray(y, dtype=np.float64).reshape(-1)
        scaled = (arr - self.min_) / self.data_range_
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)

    def inverse_transform(self, y_scaled: NDArray[np.floating]) -> NDArray[np.float64]:
        self._check_is_fitted()
        arr = np.asarray(y_scaled, dtype=np.float64).reshape(-1)
        return arr * self.data_range_ + self.min_

    def _check_is_fitted(self) -> None:
        if not self.fitted_:
            msg = "MinMaxScaler1D must be fitted before calling transform."
            raise RuntimeError(msg)


__all__: tuple[str, ...] = ("MinMaxScaler1D", "StandardScaler")
