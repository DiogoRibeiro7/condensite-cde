from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - depends on runner env
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

pytestmark = pytest.mark.unit


def test_point_widths_sum_to_range() -> None:
    grid = np.linspace(-2.0, 2.0, 11)
    widths = CondensiteTorchCDE._point_widths_numpy(grid)
    assert np.all(widths > 0)
    assert np.isclose(widths.sum(), grid[-1] - grid[0], atol=1e-6)


def test_pdf_normalization_preserves_unit_mass() -> None:
    config = CondensiteTorchCDEConfig(hidden_sizes=(4,), m_aux=4)
    estimator = CondensiteTorchCDE(config=config, random_seed=0)
    grid = np.linspace(-1.0, 1.0, 6).astype(np.float32)
    torch.manual_seed(0)
    raw = torch.rand((3, grid.size, len(estimator._bandwidths)), dtype=torch.float32)
    normalized = estimator._normalize_pdf_heads(raw, torch.from_numpy(grid))
    weights = CondensiteTorchCDE._point_widths_numpy(grid.astype(np.float64))
    mass = np.sum(normalized.detach().cpu().numpy() * weights[None, :, None], axis=1)
    assert np.allclose(mass, 1.0, atol=1e-6)
