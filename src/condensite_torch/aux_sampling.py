"""Auxiliary y' sampling strategies used during Condensite training."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

_FIXED_GRIDS: dict[tuple[int, float, float], Tensor] = {}
_EXPECTED_SHAPE_LEN = 2


def _ensure_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != _EXPECTED_SHAPE_LEN:
        msg = f"Expected (batch, m_aux) shape, got {shape}"
        raise ValueError(msg)
    batch, m_aux = shape
    if batch <= 0 or m_aux <= 0:
        msg = "shape entries must be positive"
        raise ValueError(msg)
    return batch, m_aux


def _make_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def _uniform_base_sample(method: str, shape: tuple[int, int], seed: int | None) -> Tensor:
    batch, m_aux = _ensure_shape(shape)
    generator = _make_generator(seed)
    method_lower = method.lower()

    if method_lower == "iid":
        return torch.rand((batch, m_aux), generator=generator)

    if method_lower == "stratified":
        bins = torch.arange(m_aux, dtype=torch.float32).repeat(batch, 1)
        jitter = torch.rand((batch, m_aux), generator=generator)
        return (bins + jitter) / float(m_aux)

    if method_lower == "sobol":
        sobol_seed = seed if seed is not None else 0
        engine = SobolEngine(dimension=m_aux, scramble=True, seed=sobol_seed)  # type: ignore[no-untyped-call]
        return engine.draw(batch)

    if method_lower == "fixed_grid":
        key = (m_aux, 0.0, 1.0)
        if key not in _FIXED_GRIDS:
            _FIXED_GRIDS[key] = torch.linspace(0.0, 1.0, steps=m_aux).unsqueeze(0)
        grid = _FIXED_GRIDS[key]
        return grid.repeat(batch, 1)

    msg = f"Unknown sampling method '{method}'."
    raise ValueError(msg)


def sample_yprime(
    method: str,
    shape: tuple[int, int],
    seed: int | None = None,
    device: torch.device | str | None = None,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> Tensor:
    """Sample auxiliary y' points deterministically when a seed is provided."""
    low, high = value_range
    if high <= low:
        msg = "value_range must satisfy high > low"
        raise ValueError(msg)
    base = _uniform_base_sample(method, shape, seed)
    scaled = low + (high - low) * base
    if device is None:
        return scaled
    return scaled.to(device=device)


__all__ = ("sample_yprime",)
