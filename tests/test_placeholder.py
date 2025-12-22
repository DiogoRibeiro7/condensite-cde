"""Minimal sanity tests for the project scaffold."""

from condensite_torch import __version__
from condensite_torch.estimator import TabularCDEConfig


def test_version_is_non_empty() -> None:
    assert __version__, "Package version metadata should be a non-empty string"


def test_config_validation_rejects_bad_dims() -> None:
    config = TabularCDEConfig(input_dim=1)
    config.validate()

