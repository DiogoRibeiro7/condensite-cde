"""Input validation utilities for Condensite estimators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

_EXPECTED_FEATURE_DIM = 2


@dataclass(slots=True)
class SchemaConstraints:
    """Rules describing expected data types, cardinalities, and bounds."""

    numeric_indices: Sequence[int] | None = None
    categorical_indices: Sequence[int] | None = None
    max_categorical_cardinality: int | None = None
    allow_missing_numeric: bool = True
    allow_missing_categorical: bool = True
    y_min: float | None = None
    y_max: float | None = None


class ValidationError(ValueError):
    """Raised when inputs fail schema validation."""


def validate_inputs(
    X: NDArray[Any],
    y: NDArray[Any] | None = None,
    *,
    schema: SchemaConstraints | None = None,
    context: str = "fit",
) -> None:
    """Validate tabular inputs against a schema and safety rules."""
    arr = np.asarray(X, dtype=object)
    if arr.ndim != _EXPECTED_FEATURE_DIM:
        msg = f"{context}: expected a 2-D feature matrix, got shape {arr.shape}."
        raise ValidationError(msg)
    n_rows = arr.shape[0]
    if y is not None:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            msg = f"{context}: targets must be 1-D, received {y_arr.shape}."
            raise ValidationError(msg)
        if y_arr.shape[0] != n_rows:
            msg = f"{context}: X rows ({n_rows}) must match y rows ({y_arr.shape[0]})."
            raise ValidationError(msg)
        _validate_targets(y_arr, schema)

    _validate_numeric_infinities(arr, schema)
    if schema is None:
        return
    _validate_missingness(arr, schema)
    _validate_categorical_cardinality(arr, schema)


def _validate_targets(values: NDArray[Any], schema: SchemaConstraints | None) -> None:
    """Validate target finiteness and optional user-supplied bounds."""
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = "Targets must be numeric."
        raise ValidationError(msg) from exc
    if not np.all(np.isfinite(numeric)):
        msg = "Targets contain NaN or inf values; please clean the array before training."
        raise ValidationError(msg)
    if schema is None:
        return
    if schema.y_min is not None and np.any(numeric < schema.y_min - 1e-12):
        msg = f"Targets have values below {schema.y_min}; check scaling assumptions."
        raise ValidationError(msg)
    if schema.y_max is not None and np.any(numeric > schema.y_max + 1e-12):
        msg = f"Targets exceed {schema.y_max}; check scaling assumptions."
        raise ValidationError(msg)


def _validate_numeric_infinities(
    arr: NDArray[Any],
    schema: SchemaConstraints | None,
) -> None:
    """Reject infinities in numeric columns while leaving NaN to missingness policy."""
    if schema is not None and schema.numeric_indices is not None:
        indices = list(schema.numeric_indices)
    else:
        indices = list(range(arr.shape[1]))

    for idx in indices:
        column = _safe_column(arr, idx)
        try:
            numeric = np.asarray(column, dtype=np.float64)
        except (TypeError, ValueError):
            # Without an explicit numeric schema this is simply a categorical column.
            if schema is not None and schema.numeric_indices is not None:
                msg = f"Numeric column {idx} contains non-numeric values."
                raise ValidationError(msg) from None
            continue
        if np.any(np.isinf(numeric)):
            msg = f"Numeric column {idx} contains infinite values."
            raise ValidationError(msg)


def _validate_missingness(arr: NDArray[Any], schema: SchemaConstraints) -> None:
    """Ensure missingness conforms to schema opts."""
    if schema.numeric_indices is not None:
        for idx in schema.numeric_indices:
            column = _safe_column(arr, idx)
            mask = _is_missing(column)
            if not schema.allow_missing_numeric and np.any(mask):
                msg = f"Numeric column {idx} has missing values but allow_missing_numeric=False."
                raise ValidationError(msg)
    if schema.categorical_indices is not None:
        for idx in schema.categorical_indices:
            column = _safe_column(arr, idx)
            mask = _is_missing(column)
            if not schema.allow_missing_categorical and np.any(mask):
                msg = (
                    f"Categorical column {idx} has missing values but "
                    "allow_missing_categorical=False."
                )
                raise ValidationError(msg)


def _validate_categorical_cardinality(arr: NDArray[Any], schema: SchemaConstraints) -> None:
    """Raise when categorical columns exceed allowed unique levels."""
    if schema.max_categorical_cardinality is None or schema.categorical_indices is None:
        return
    max_card = int(schema.max_categorical_cardinality)
    for idx in schema.categorical_indices:
        column = _safe_column(arr, idx)
        normalized = np.asarray(column, dtype=object)
        values = normalized[~_is_missing(normalized)]
        unique_count = np.unique(values).size
        if unique_count > max_card:
            msg = (
                f"Categorical column {idx} has {unique_count} unique values, "
                f"exceeding the limit of {max_card}. Consider marking it numeric or hashing."
            )
            raise ValidationError(msg)


def _safe_column(arr: NDArray[Any], idx: int) -> NDArray[Any]:
    """Return a 1-D column with bounds checking."""
    if idx < 0 or idx >= arr.shape[1]:
        msg = f"Column index {idx} is out of bounds for {arr.shape[1]} features."
        raise ValidationError(msg)
    return arr[:, idx]


def _is_missing(values: NDArray[Any]) -> NDArray[np.bool_]:
    """Return boolean mask for missing entries supporting object dtypes."""
    if values.dtype == object:
        mask = np.array(
            [val is None or (isinstance(val, float) and np.isnan(val)) for val in values],
        )
        return mask
    return np.isnan(values.astype(np.float64, copy=False))


__all__ = ("SchemaConstraints", "ValidationError", "validate_inputs")
