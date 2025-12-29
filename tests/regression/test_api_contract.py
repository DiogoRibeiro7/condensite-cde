from __future__ import annotations

import inspect

import pytest

from condensite_torch import CondensiteTorchCDE

pytestmark = pytest.mark.regression


def _has_parameter(signature: inspect.Signature, name: str) -> bool:
    return name in signature.parameters


def test_fit_signature_contains_expected_args() -> None:
    sig = inspect.signature(CondensiteTorchCDE.fit)
    for name in ("X", "y"):
        assert _has_parameter(sig, name)


def test_predict_density_signature() -> None:
    sig = inspect.signature(CondensiteTorchCDE.predict_density)
    for name in ("X", "y_grid"):
        assert _has_parameter(sig, name)


def test_predict_cdf_signature() -> None:
    sig = inspect.signature(CondensiteTorchCDE.predict_cdf)
    for name in ("X", "y_grid"):
        assert _has_parameter(sig, name)


def test_sample_signature_includes_n_samples() -> None:
    sig = inspect.signature(CondensiteTorchCDE.sample)
    assert _has_parameter(sig, "n_samples")
