"""Auxiliary y' sampling strategies used during Condensite training."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.quasirandom import SobolEngine

_FIXED_GRIDS: dict[tuple[int, float, float], Tensor] = {}
_EXPECTED_SHAPE_LEN = 2
_IMPORTANCE_EPS = 1e-6
FloatArray = NDArray[np.float64]


def _ensure_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Validate `(batch, m_aux)` shapes supplied to samplers.

    Args:
        shape (tuple[int, int]): Requested dimensions.

    Returns:
        tuple[int, int]: Same shape after validation.

    Raises:
        ValueError: If length is not 2 or entries are non-positive.

    Side Effects:
        None.

    Complexity:
        O(1).
    """
    if len(shape) != _EXPECTED_SHAPE_LEN:
        msg = f"Expected (batch, m_aux) shape, got {shape}"
        raise ValueError(msg)
    batch, m_aux = shape
    if batch <= 0 or m_aux <= 0:
        msg = "shape entries must be positive"
        raise ValueError(msg)
    return batch, m_aux


def _make_generator(seed: int | None) -> torch.Generator | None:
    """Create a CPU generator when a deterministic seed is provided.

    Args:
        seed (int | None): Seed to use; if `None`, return `None`.

    Returns:
        torch.Generator | None: CPU generator seeded with `seed`.

    Raises:
        None.

    Side Effects:
        None.

    Complexity:
        O(1).
    """
    if seed is None:
        return None
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def _uniform_base_sample(method: str, shape: tuple[int, int], seed: int | None) -> Tensor:
    """Draw base samples in [0,1] according to the requested method.

    Args:
        method (str): Sampling strategy (`iid`, `stratified`, `lhs`, `sobol`, `fixed_grid`).
        shape (tuple[int, int]): `(batch, m_aux)` shape for the samples.
        seed (int | None): Optional deterministic seed.

    Returns:
        Tensor: Samples shaped `[batch, m_aux]` in `[0,1]`.

    Raises:
        ValueError: If method name is unknown.

    Side Effects:
        Updates cached fixed grids for the deterministic mode.

    Complexity:
        O(batch * m_aux).
    """
    batch, m_aux = _ensure_shape(shape)
    generator = _make_generator(seed)
    method_lower = method.lower()

    if method_lower == "iid":
        return torch.rand((batch, m_aux), generator=generator)

    if method_lower == "stratified":
        bins = torch.arange(m_aux, dtype=torch.float32).repeat(batch, 1)
        jitter = torch.rand((batch, m_aux), generator=generator)
        return (bins + jitter) / float(m_aux)

    if method_lower == "lhs":
        bins = torch.arange(m_aux, dtype=torch.float32).repeat(batch, 1)
        jitter = torch.rand((batch, m_aux), generator=generator)
        base = (bins + jitter) / float(m_aux)
        permuted = torch.empty_like(base)
        for row in range(batch):
            order = torch.randperm(m_aux, generator=generator)
            permuted[row] = base[row, order]
        return permuted

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


class ImportanceSampler:
    """Piecewise-uniform importance sampler built from empirical targets."""

    def __init__(self, bin_edges: FloatArray, bin_probs: FloatArray) -> None:
        """Instantiate the sampler with explicit histogram bins.

        Args:
            bin_edges (np.ndarray): Monotone edges with length `n_bins + 1`.
            bin_probs (np.ndarray): Probabilities per bin.

        Returns:
            None.

        Raises:
            ValueError: If dimensions mismatch or lengths are invalid.

        Side Effects:
            Materializes `torch.Tensor` copies for runtime sampling.
        """
        if bin_edges.ndim != 1:
            msg = "bin_edges must be 1-D."
            raise ValueError(msg)
        if bin_probs.ndim != 1:
            msg = "bin_probs must be 1-D."
            raise ValueError(msg)
        if bin_edges.size != bin_probs.size + 1:
            msg = "bin_edges must have len(probs) + 1 entries."
            raise ValueError(msg)
        probs = bin_probs.astype(np.float32, copy=False)
        probs = np.clip(probs, _IMPORTANCE_EPS, None)
        probs /= probs.sum()
        edges = bin_edges.astype(np.float32, copy=False)
        widths = np.diff(edges)
        widths = np.clip(widths, _IMPORTANCE_EPS, None)
        pdf = probs / widths
        cdf = np.cumsum(probs)

        self._bin_lefts = torch.from_numpy(edges[:-1])
        self._bin_widths = torch.from_numpy(widths)
        self._bin_probs = torch.from_numpy(probs)
        self._bin_pdf = torch.from_numpy(pdf)
        self._cdf = torch.from_numpy(cdf)

    @classmethod
    def from_array(
        cls,
        values: FloatArray,
        *,
        bins: int = 64,
        tail_bonus: float = 0.05,
    ) -> ImportanceSampler:
        """Build a sampler by histogramming empirical values.

        Args:
            values (np.ndarray): Raw values normalized to `[0,1]`.
            bins (int): Number of histogram bins.
            tail_bonus (float): Extra pseudo-count mass for edge bins.

        Returns:
            ImportanceSampler: Sampler ready for `.draw`.

        Raises:
            ValueError: If `values` is empty.

        Side Effects:
            None beyond `__init__`.

        Complexity:
            O(n + bins).
        """
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError("values must contain at least one entry")
        clipped = np.clip(arr, 0.0, 1.0)
        counts, edges = np.histogram(clipped, bins=bins, range=(0.0, 1.0))
        counts = counts.astype(np.float64)
        tail_mass = float(arr.size) * tail_bonus
        counts[0] += tail_mass
        counts[-1] += tail_mass
        counts += _IMPORTANCE_EPS
        probs = counts / counts.sum()
        return cls(edges, probs)

    def draw(
        self,
        shape: tuple[int, int],
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draw importance-sampled coordinates plus weights.

        Args:
            shape (tuple[int, int]): `(batch, m_aux)` size.
            seed (int | None): Optional deterministic seed.
            device (torch.device | str | None): Device for the returned tensors.

        Returns:
            tuple[Tensor, Tensor]: `(samples, weights)` both shaped `[batch, m_aux]`.

        Raises:
            ValueError: If `shape` is invalid.

        Side Effects:
            None.

        Complexity:
            O(batch * m_aux).
        """
        _ensure_shape(shape)
        generator = _make_generator(seed)
        base = torch.rand(shape, generator=generator, dtype=torch.float32)
        base_flat = base.reshape(-1)
        cdf = self._cdf.to(base_flat.device)
        idx = torch.bucketize(base_flat, cdf, right=False)
        probs = self._bin_probs.to(base_flat.device)[idx]
        starts = torch.where(
            idx == 0,
            torch.zeros_like(base_flat),
            cdf[idx - 1],
        )
        local = torch.where(
            probs > 0,
            (base_flat - starts) / probs,
            torch.zeros_like(base_flat),
        )
        lefts = self._bin_lefts.to(base_flat.device)[idx]
        widths = self._bin_widths.to(base_flat.device)[idx]
        samples = lefts + local * widths
        pdf = self._bin_pdf.to(base_flat.device)[idx]
        # Weight inversely proportional to proposal pdf to preserve unbiased targets.
        weights = torch.where(pdf > 0, 1.0 / pdf, torch.ones_like(pdf))
        weights /= torch.mean(weights)
        samples = samples.reshape(shape)
        weights = weights.reshape(shape)
        if device is not None:
            samples = samples.to(device=device)
            weights = weights.to(device=device)
        return samples, weights


def sample_yprime(
    method: str,
    shape: tuple[int, int],
    seed: int | None = None,
    device: torch.device | str | None = None,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> Tensor:
    """Sample auxiliary y' points deterministically when a seed is provided.

    Args:
        method (str): Sampling strategy identifier.
        shape (tuple[int, int]): `(batch, m_aux)` shape.
        seed (int | None): RNG seed for determinism.
        device (torch.device | str | None): Desired output device.
        value_range (tuple[float, float]): Inclusive range to scale samples into.

    Returns:
        Tensor: Samples shaped `[batch, m_aux]`.

    Raises:
        ValueError: If `value_range` is invalid or method is unknown.

    Side Effects:
        None.

    Complexity:
        O(batch * m_aux).
    """
    low, high = value_range
    if high <= low:
        msg = "value_range must satisfy high > low"
        raise ValueError(msg)
    base = _uniform_base_sample(method, shape, seed)
    scaled = low + (high - low) * base
    if device is None:
        return scaled
    return scaled.to(device=device)


__all__ = ("ImportanceSampler", "sample_yprime")
