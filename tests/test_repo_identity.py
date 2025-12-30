"""Ensure repo metadata stays aligned with the canonical name."""

from __future__ import annotations

from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    tomllib = None  # type: ignore[assignment]

REPO_NAME = "condensite-cde"
PYPROJECT_PATH = Path("pyproject.toml")
README_PATH = Path("README.md")


def _load_pyproject_name() -> str:
    if tomllib is not None:
        data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        project = data.get("project", {})
        name = project.get("name")
        if isinstance(name, str):
            return name
    # Minimal fallback parser looking inside the [project] section.
    in_project = False
    split_expected = 2
    for raw_line in PYPROJECT_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            in_project = line == "[project]"
            continue
        if in_project and line.startswith("name"):
            parts = line.split("=", maxsplit=1)
            if len(parts) == split_expected:
                return parts[1].strip().strip('"').strip("'")
    msg = "Could not determine package name from pyproject.toml"
    raise AssertionError(msg)


def test_pyproject_has_expected_name() -> None:
    assert _load_pyproject_name() == REPO_NAME


def test_readme_mentions_repo_name() -> None:
    readme_text = README_PATH.read_text(encoding="utf-8")
    assert REPO_NAME in readme_text
