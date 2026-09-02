from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - depends on host torch install
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch.kernels import (
    epanechnikov_kernel_np,
    get_kernel_spec,
    kernel_h_np,
    kernel_h_torch,
)

pytestmark = pytest.mark.unit


def test_kernel_bandwidth_must_be_positive() -> None:
    y = np.array([0.0])
    with pytest.raises(ValueError):
        kernel_h_np(y, y, bandwidth=0.0)


@pytest.mark.parametrize("bandwidth", [0.1, 1.0])
def test_gaussian_kernel_outputs_are_finite_np(bandwidth: float) -> None:
    y = np.array([[0.1, -0.2], [0.5, 0.3]])
    y_prime = np.zeros_like(y)
    values = kernel_h_np(y, y_prime, bandwidth)
    assert values.shape == y.shape
    assert np.all(values >= 0.0)
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize("bandwidth", [0.1, 0.7])
def test_torch_kernel_matches_numpy(bandwidth: float) -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=(3, 4)).astype(np.float32)
    y_prime = rng.normal(size=(3, 4)).astype(np.float32)
    expected = kernel_h_np(y, y_prime, bandwidth)
    torch_vals = kernel_h_torch(torch.from_numpy(y), torch.from_numpy(y_prime), bandwidth).numpy()
    assert torch_vals.shape == expected.shape
    assert np.allclose(torch_vals, expected, atol=1e-6)


@pytest.mark.parametrize("bandwidth", [0.2, 0.8])
def test_gaussian_kernel_torch_tensor_bandwidth(bandwidth: float) -> None:
    y = torch.randn(2, 3)
    y_prime = torch.randn(2, 3)
    bw = torch.full_like(y, bandwidth)
    result = kernel_h_torch(y, y_prime, bw)
    assert result.shape == y.shape
    assert torch.all(result >= 0.0)
    assert torch.isfinite(result).all()


def test_epanechnikov_kernel_is_nonnegative_and_integrates_to_one() -> None:
    bandwidth = 0.4
    delta = np.linspace(-2 * bandwidth, 2 * bandwidth, 4000)
    values = epanechnikov_kernel_np(delta, bandwidth)
    negative_tol = 1e-10
    support_margin = 1e-12
    integral_tol = 1e-2
    assert np.all(values >= -negative_tol)
    assert np.all(values[np.abs(delta) > bandwidth + support_margin] == pytest.approx(0.0))
    integral = np.trapezoid(values, delta)
    assert abs(integral - 1.0) < integral_tol


def test_kernel_registry_returns_callable_specs() -> None:
    spec = get_kernel_spec("epanechnikov")
    y = torch.tensor([[0.1, -0.2]])
    y_prime = torch.zeros_like(y)
    result = spec.torch_fn(y, y_prime, 0.5)
    assert result.shape == y.shape
    assert torch.all(result >= 0.0)
    with pytest.raises(ValueError):
        get_kernel_spec("does-not-exist")
