from __future__ import annotations

import numpy as np
import pytest

from condensite_torch.preprocessing import TabularPreprocessor, TabularPreprocessorConfig

pytestmark = pytest.mark.unit


def _make_dataset() -> np.ndarray:
    return np.array(
        [
            [1.0, "red", "cat"],
            [np.nan, "blue", None],
            [3.5, "red", "dog"],
            [2.1, None, "cat"],
        ],
        dtype=object,
    )


def test_preprocessor_imputes_missing_and_orders_features() -> None:
    data = _make_dataset()
    config = TabularPreprocessorConfig(add_missing_indicator=True)
    preprocessor = TabularPreprocessor(config).fit(data)
    transformed = preprocessor.transform(data)
    expected_feature_count = 10
    assert transformed.shape == (data.shape[0], expected_feature_count)
    assert not np.isnan(transformed).any()
    assert preprocessor.feature_names_ == [
        "x0",
        "x0_missing",
        "x1=red",
        "x1=blue",
        "x1=__missing__",
        "x1=__unknown__",
        "x2=cat",
        "x2=dog",
        "x2=__missing__",
        "x2=__unknown__",
    ]


def test_unknown_category_raises_when_configured() -> None:
    data = _make_dataset()
    config = TabularPreprocessorConfig(handle_unknown="error")
    preprocessor = TabularPreprocessor(config).fit(data)
    new_sample = np.array([[5.0, "green", "cat"]], dtype=object)
    with pytest.raises(ValueError):
        preprocessor.transform(new_sample)


def test_unknown_category_maps_to_bucket_when_enabled() -> None:
    data = _make_dataset()
    config = TabularPreprocessorConfig(handle_unknown="use_unknown")
    preprocessor = TabularPreprocessor(config).fit(data)
    new_sample = np.array([[5.0, "green", "cat"]], dtype=object)
    transformed = preprocessor.transform(new_sample)
    assert transformed.shape[1] == len(preprocessor.feature_names_)
    # ensure unknown category triggered bucket
    assert transformed[0, preprocessor.feature_names_.index("x1=__unknown__")] == pytest.approx(1.0)


def test_preprocessor_round_trip_serialization() -> None:
    data = _make_dataset()
    preprocessor = TabularPreprocessor().fit(data)
    payload = preprocessor.to_dict()
    restored = TabularPreprocessor.from_dict(payload)
    original = preprocessor.transform(data)
    round_trip = restored.transform(data)
    assert np.allclose(original, round_trip)
