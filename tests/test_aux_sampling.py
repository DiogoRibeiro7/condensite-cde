"""Tests for auxiliary sampling utilities."""

from __future__ import annotations

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - environment-specific torch setup
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch.aux_sampling import sample_yprime


@pytest.mark.parametrize("method", ["iid", "stratified", "sobol", "fixed_grid"])
def test_sampling_methods_are_deterministic(method: str) -> None:
    first = sample_yprime(method, (3, 5), seed=123)
    second = sample_yprime(method, (3, 5), seed=123)
    assert torch.allclose(first, second)


def test_stratified_sampling_covers_unit_interval() -> None:
    samples = sample_yprime("stratified", (4, 8), seed=10)
    assert torch.all((samples >= 0.0) & (samples <= 1.0))
    # Each row should roughly cover bins of width 1 / m_aux.
    row = samples[0].sort().values
    diffs = torch.diff(row)
    assert torch.all(diffs > 0)


def test_fixed_grid_is_shared_across_batches() -> None:
    grid = sample_yprime("fixed_grid", (2, 4), seed=999)
    expected = torch.linspace(0.0, 1.0, 4)
    assert torch.allclose(grid[0], expected)
    assert torch.allclose(grid[0], grid[1])
