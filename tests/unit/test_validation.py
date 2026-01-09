"""Unit tests for input validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from condensite_torch.validation import SchemaConstraints, ValidationError, validate_inputs


def test_validate_rejects_non_2d_features() -> None:
    X = np.array([1.0, 2.0])
    with pytest.raises(ValidationError, match="2-D feature matrix"):
        validate_inputs(X, None)


def test_validate_enforces_target_bounds() -> None:
    X = np.zeros((3, 1))
    y = np.array([-0.1, 0.2, 0.3])
    schema = SchemaConstraints(y_min=0.0, y_max=1.0)
    with pytest.raises(ValidationError, match=r"below 0\.0"):
        validate_inputs(X, y, schema=schema)


def test_validate_missing_numeric_not_allowed() -> None:
    X = np.array([[0.0, np.nan], [1.0, 2.0]], dtype=object)
    schema = SchemaConstraints(numeric_indices=[1], allow_missing_numeric=False)
    with pytest.raises(ValidationError, match="missing values"):
        validate_inputs(X, None, schema=schema)


def test_validate_categorical_cardinality_limit() -> None:
    X = np.array([["a"], ["b"], ["c"], ["d"]], dtype=object)
    schema = SchemaConstraints(categorical_indices=[0], max_categorical_cardinality=2)
    with pytest.raises(ValidationError, match="exceeding the limit"):
        validate_inputs(X, None, schema=schema)


def test_validate_accepts_clean_inputs() -> None:
    X = np.array([[0.0, "a"], [1.0, "b"]], dtype=object)
    y = np.array([0.0, 1.0])
    schema = SchemaConstraints(
        numeric_indices=[0],
        categorical_indices=[1],
        allow_missing_numeric=False,
        allow_missing_categorical=False,
        max_categorical_cardinality=4,
        y_min=0.0,
        y_max=1.0,
    )
    validate_inputs(X, y, schema=schema)
