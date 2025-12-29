"""Minimal sanity tests for the project scaffold."""

from condensite_torch import __version__


def test_version_is_non_empty() -> None:
    assert __version__, "Package version metadata should be a non-empty string"

