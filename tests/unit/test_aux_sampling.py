from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - environment-specific torch availability
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import sample_yprime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", ["iid", "stratified", "sobol", "fixed_grid"])
def test_sampling_methods_are_bounded_and_deterministic(method: str) -> None:
    samples_a = sample_yprime(method, (3, 5), seed=123)
    samples_b = sample_yprime(method, (3, 5), seed=123)
    assert samples_a.shape == (3, 5)
    assert torch.allclose(samples_a, samples_b)
    assert torch.all((samples_a >= 0.0) & (samples_a <= 1.0))


def test_stratified_sampling_covers_bins() -> None:
    m_aux = 8
    samples = sample_yprime("stratified", (1, m_aux), seed=0)[0].numpy()
    bins = np.floor(samples * m_aux).astype(int)
    assert set(bins) == set(range(m_aux))


def test_fixed_grid_is_deterministic_without_seed() -> None:
    grid = sample_yprime("fixed_grid", (2, 4), seed=None)
    expected = torch.linspace(0.0, 1.0, steps=4).repeat(2, 1)
    assert torch.allclose(grid, expected)
