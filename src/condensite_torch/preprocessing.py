"""Tabular preprocessing utilities for numeric + categorical features."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

CATEGORICAL_ENCODING = Literal["onehot"]
UNKNOWN_HANDLING = Literal["error", "use_unknown"]
ObjectArray = NDArray[np.object_]
FloatArray = NDArray[np.float64]

_EXPECTED_INPUT_DIM = 2


@dataclass(slots=True)
class TabularPreprocessorConfig:
    """Configuration for tabular preprocessing."""

    categorical_indices: Sequence[int] | None = None
    numeric_indices: Sequence[int] | None = None
    categorical_encoding: CATEGORICAL_ENCODING = "onehot"
    handle_unknown: UNKNOWN_HANDLING = "use_unknown"
    add_missing_indicator: bool = False


class TabularPreprocessor:
    """Impute and encode mixed tabular inputs with deterministic feature ordering."""

    def __init__(self, config: TabularPreprocessorConfig | None = None) -> None:
        self.config = config or TabularPreprocessorConfig()
        self.numeric_indices: list[int] = []
        self.categorical_indices: list[int] = []
        self.numeric_impute_: dict[int, float] = {}
        self.categorical_modes_: dict[int, str] = {}
        self.categorical_categories_: dict[int, list[str]] = {}
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit(self, X: ObjectArray) -> TabularPreprocessor:
        """Learn column roles, imputers, and categorical vocabularies."""
        arr = self._ensure_2d(X)
        n_features = arr.shape[1]
        all_indices = set(range(n_features))

        configured_numeric = (
            set(self.config.numeric_indices) if self.config.numeric_indices is not None else None
        )
        configured_categorical = (
            set(self.config.categorical_indices)
            if self.config.categorical_indices is not None
            else None
        )
        if configured_numeric is not None and configured_categorical is not None:
            overlap = configured_numeric.intersection(configured_categorical)
            if overlap:
                msg = f"Columns cannot be both numeric and categorical: {sorted(overlap)}"
                raise ValueError(msg)

        if configured_numeric is not None:
            numeric = configured_numeric
        elif configured_categorical is not None:
            numeric = all_indices.difference(configured_categorical)
        else:
            numeric = {idx for idx in range(n_features) if self._looks_numeric(arr[:, idx])}

        if configured_categorical is not None:
            categorical = configured_categorical
        else:
            categorical = all_indices.difference(numeric)

        selected = numeric.union(categorical)
        invalid = [idx for idx in selected if idx < 0 or idx >= n_features]
        if invalid:
            msg = f"Configured column indices out of bounds: {sorted(invalid)}"
            raise ValueError(msg)

        self.numeric_indices = sorted(numeric)
        self.categorical_indices = sorted(categorical)
        self._fit_numeric(arr)
        self._fit_categorical(arr)
        self.feature_names_ = self._build_feature_names()
        self._fitted = True
        return self

    def fit_transform(self, X: ObjectArray) -> FloatArray:
        """Fit preprocessing state and transform ``X``."""
        self.fit(X)
        return self.transform(X)

    def transform(self, X: ObjectArray) -> FloatArray:
        """Transform ``X`` using fitted preprocessing state."""
        if not self._fitted:
            msg = "Call fit() before transform()."
            raise RuntimeError(msg)
        arr = self._ensure_2d(X)
        parts: list[FloatArray] = []
        if self.numeric_indices:
            parts.append(self._transform_numeric(arr))
        if self.categorical_indices:
            parts.append(self._transform_categorical(arr))
        if not parts:
            msg = "No features were selected for preprocessing; provide at least one column."
            raise ValueError(msg)
        return np.hstack(parts).astype(np.float64, copy=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted preprocessing state."""
        if not self._fitted:
            msg = "Cannot serialize an unfitted preprocessor."
            raise RuntimeError(msg)
        return {
            "config": {
                "categorical_indices": list(self.config.categorical_indices or []),
                "numeric_indices": list(self.config.numeric_indices or []),
                "categorical_encoding": self.config.categorical_encoding,
                "handle_unknown": self.config.handle_unknown,
                "add_missing_indicator": self.config.add_missing_indicator,
            },
            "numeric_indices": self.numeric_indices,
            "categorical_indices": self.categorical_indices,
            "numeric_impute": self.numeric_impute_,
            "categorical_modes": self.categorical_modes_,
            "categorical_categories": self.categorical_categories_,
            "feature_names": self.feature_names_,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TabularPreprocessor:
        """Restore preprocessing state from ``to_dict`` output."""
        config_data = payload["config"]
        config = TabularPreprocessorConfig(
            categorical_indices=config_data.get("categorical_indices") or None,
            numeric_indices=config_data.get("numeric_indices") or None,
            categorical_encoding=config_data.get("categorical_encoding", "onehot"),
            handle_unknown=config_data.get("handle_unknown", "use_unknown"),
            add_missing_indicator=bool(config_data.get("add_missing_indicator", False)),
        )
        preprocessor = cls(config=config)
        preprocessor.numeric_indices = list(payload.get("numeric_indices", []))
        preprocessor.categorical_indices = list(payload.get("categorical_indices", []))
        preprocessor.numeric_impute_ = {
            int(key): float(value) for key, value in payload.get("numeric_impute", {}).items()
        }
        preprocessor.categorical_modes_ = {
            int(key): str(value) for key, value in payload.get("categorical_modes", {}).items()
        }
        preprocessor.categorical_categories_ = {
            int(key): list(value)
            for key, value in payload.get("categorical_categories", {}).items()
        }
        preprocessor.feature_names_ = list(payload.get("feature_names", []))
        preprocessor._fitted = True
        return preprocessor

    def _fit_numeric(self, arr: ObjectArray) -> None:
        self.numeric_impute_.clear()
        for idx in self.numeric_indices:
            column = self._to_float_column(arr[:, idx])
            median = float(np.nanmedian(column))
            if math.isnan(median):
                median = 0.0
            self.numeric_impute_[idx] = median

    def _fit_categorical(self, arr: ObjectArray) -> None:
        self.categorical_modes_.clear()
        self.categorical_categories_.clear()
        for idx in self.categorical_indices:
            normalized = self._normalize_categories(np.asarray(arr[:, idx], dtype=object))
            mask_valid = normalized.astype(bool)
            values = normalized[mask_valid]
            if values.size == 0:
                mode = "__missing__"
                categories = ["__missing__"]
            else:
                uniques, counts = np.unique(values, return_counts=True)
                order = np.lexsort((uniques, -counts))
                sorted_values = uniques[order]
                mode = str(sorted_values[0])
                categories = sorted_values.tolist()
                if "__missing__" not in categories and np.any(~mask_valid):
                    categories.append("__missing__")
            if self.config.handle_unknown == "use_unknown" and "__unknown__" not in categories:
                categories.append("__unknown__")
            self.categorical_modes_[idx] = mode
            self.categorical_categories_[idx] = categories

    def _transform_numeric(self, arr: ObjectArray) -> FloatArray:
        columns: list[FloatArray] = []
        for idx in self.numeric_indices:
            col = self._to_float_column(arr[:, idx])
            mask = np.isnan(col)
            imputed = np.where(mask, self.numeric_impute_[idx], col)
            columns.append(imputed.reshape(-1, 1))
            if self.config.add_missing_indicator:
                columns.append(mask.astype(np.float64).reshape(-1, 1))
        return np.hstack(columns) if columns else np.empty((arr.shape[0], 0), dtype=np.float64)

    def _transform_categorical(self, arr: ObjectArray) -> FloatArray:
        rows: list[FloatArray] = []
        for idx in self.categorical_indices:
            normalized = self._normalize_categories(np.asarray(arr[:, idx], dtype=object))
            mask_present = normalized.astype(bool)
            missing_fill = (
                "__missing__"
                if "__missing__" in self.categorical_categories_[idx]
                else self.categorical_modes_[idx]
            )
            normalized = np.where(mask_present, normalized, missing_fill)
            categories = self.categorical_categories_[idx]
            mapping = {cat: pos for pos, cat in enumerate(categories)}
            encoded = np.zeros((arr.shape[0], len(categories)), dtype=np.float64)
            for row_idx, value in enumerate(normalized):
                key = str(value)
                if key in mapping:
                    target = mapping[key]
                elif self.config.handle_unknown == "use_unknown" and "__unknown__" in mapping:
                    target = mapping["__unknown__"]
                else:
                    msg = f"Encountered unknown category '{key}' in column {idx}."
                    raise ValueError(msg)
                encoded[row_idx, target] = 1.0
            rows.append(encoded)
        return np.hstack(rows) if rows else np.empty((arr.shape[0], 0), dtype=np.float64)

    def _build_feature_names(self) -> list[str]:
        names: list[str] = []
        for idx in self.numeric_indices:
            base = f"x{idx}"
            names.append(base)
            if self.config.add_missing_indicator:
                names.append(f"{base}_missing")
        for idx in self.categorical_indices:
            for cat in self.categorical_categories_.get(idx, []):
                names.append(f"x{idx}={cat}")
        return names

    @staticmethod
    def _ensure_2d(X: ObjectArray | Sequence[Sequence[Any]]) -> ObjectArray:
        arr = np.asarray(X, dtype=object)
        if arr.ndim != _EXPECTED_INPUT_DIM:
            msg = f"Input must be 2-D, got shape {arr.shape}"
            raise ValueError(msg)
        return arr

    @staticmethod
    def _normalize_categories(values: ObjectArray) -> NDArray[np.str_]:
        normalized = np.empty(values.shape[0], dtype=object)
        for idx, value in enumerate(values):
            if value is None:
                normalized[idx] = ""
            else:
                try:
                    normalized[idx] = "" if isinstance(value, float) and np.isnan(value) else str(value)
                except TypeError:
                    normalized[idx] = ""
        return normalized.astype(str)

    @staticmethod
    def _to_float_column(values: NDArray[Any]) -> FloatArray:
        try:
            column = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            msg = "Numeric column contains non-numeric values."
            raise ValueError(msg) from exc
        if np.any(np.isinf(column)):
            msg = "Numeric column contains infinite values."
            raise ValueError(msg)
        return column

    @staticmethod
    def _looks_numeric(column: NDArray[Any]) -> bool:
        try:
            values = np.asarray(column, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        return not np.any(np.isinf(values))


__all__ = ("TabularPreprocessor", "TabularPreprocessorConfig")
