"""Condensite Torch tabular conditional density estimation utilities."""

from importlib import metadata
from typing import Final

PACKAGE_NAME: Final = "condensite-torch"

try:
    __version__ = metadata.version(PACKAGE_NAME)
except metadata.PackageNotFoundError:  # pragma: no cover - version only set when installed
    __version__ = "0.0.0"

__all__ = ["__version__", "PACKAGE_NAME"]
