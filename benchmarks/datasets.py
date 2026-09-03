from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

OUTLIER_PROB = 0.05


@dataclass(slots=True)
class DatasetBundle:
    """Container for train/test splits used by benchmark runners."""

    X_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.float64]


def _split(
    generator: np.random.Generator,
    builder: Callable[[np.random.Generator, int], tuple[NDArray[np.float64], NDArray[np.float64]]],
    n_train: int,
    n_test: int,
) -> DatasetBundle:
    """Create deterministic train/test splits for a dataset builder.

    Args:
        generator (np.random.Generator): RNG seeded per dataset.
        builder (Callable): Callable returning `(X, y)` arrays when invoked with `(rng, n)`.
        n_train (int): Number of training samples to synthesize.
        n_test (int): Number of test samples to synthesize.

    Returns:
        DatasetBundle: Object with separate train/test arrays.

    Raises:
        None.

    Side Effects:
        Consumes random numbers from `generator`.

    Complexity:
        O(n_train + n_test).
    """
    X_train, y_train = builder(generator, n_train)
    X_test, y_test = builder(generator, n_test)
    return DatasetBundle(X_train, y_train, X_test, y_test)


def heteroscedastic(n_train: int = 400, n_test: int = 200, seed: int = 0) -> DatasetBundle:
    """Sample a 2D heteroscedastic regression problem.

    Args:
        n_train (int): Number of training rows.
        n_test (int): Number of test rows.
        seed (int): Seed controlling NumPy's generator.

    Returns:
        DatasetBundle: Features and targets with input-dependent noise.

    Raises:
        None.

    Side Effects:
        None.

    Complexity:
        O(n_train + n_test).
    """

    def builder(
        rng: np.random.Generator,
        n: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = rng.uniform(-2.0, 2.0, size=(n, 2))
        noise_scale = 0.1 + 0.35 * (np.sin(X[:, 0]) ** 2)
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + noise_scale * rng.normal(size=n)
        return X, y

    rng = np.random.default_rng(seed)
    return _split(rng, builder, n_train, n_test)


def multimodal(n_train: int = 400, n_test: int = 200, seed: int = 1) -> DatasetBundle:
    """Sample a bimodal target conditional on 2D Gaussian features.

    Args:
        n_train (int): Number of training rows.
        n_test (int): Number of test rows.
        seed (int): RNG seed.

    Returns:
        DatasetBundle: Bimodal targets with Gaussian covariates.

    Raises:
        None.

    Side Effects:
        None.

    Complexity:
        O(n_train + n_test).
    """

    def builder(
        rng: np.random.Generator,
        n: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = rng.normal(size=(n, 2))
        mode_choice = rng.integers(0, 2, size=n)
        base = np.sin(2 * X[:, 0]) + 0.25 * X[:, 1]
        y = np.where(mode_choice == 0, base + 0.5, -base - 0.5)
        y += 0.2 * rng.normal(size=n)
        return X, y

    rng = np.random.default_rng(seed)
    return _split(rng, builder, n_train, n_test)


def heavy_tail(n_train: int = 400, n_test: int = 200, seed: int = 2) -> DatasetBundle:
    """Sample a Student-t noise problem that stresses tail behaviour.

    Args:
        n_train (int): Number of training rows.
        n_test (int): Number of test rows.
        seed (int): RNG seed.

    Returns:
        DatasetBundle: Heavy-tailed targets with linear trend.

    Raises:
        None.

    Side Effects:
        None.

    Complexity:
        O(n_train + n_test).
    """

    def builder(
        rng: np.random.Generator,
        n: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = rng.normal(size=(n, 2))
        noise = 0.2 * rng.standard_t(df=3, size=n)
        y = 0.8 * X[:, 0] - 0.3 * X[:, 1] + noise
        return X, y

    rng = np.random.default_rng(seed)
    return _split(rng, builder, n_train, n_test)


def skewed(n_train: int = 400, n_test: int = 200, seed: int = 3) -> DatasetBundle:
    """Sample skewed targets using log-normal noise.

    Args:
        n_train (int): Number of training rows.
        n_test (int): Number of test rows.
        seed (int): RNG seed.

    Returns:
        DatasetBundle: Dataset with asymmetric targets.

    Raises:
        None.

    Side Effects:
        None.

    Complexity:
        O(n_train + n_test).
    """

    def builder(
        rng: np.random.Generator,
        n: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = rng.normal(size=(n, 2))
        skew_noise = rng.lognormal(mean=0.0, sigma=0.3, size=n) - 1.0
        y = 0.5 * np.sin(X[:, 0]) + 0.3 * X[:, 1] + 0.2 * skew_noise
        return X, y

    rng = np.random.default_rng(seed)
    return _split(rng, builder, n_train, n_test)


def outliers(n_train: int = 400, n_test: int = 200, seed: int = 4) -> DatasetBundle:
    """Sample mostly-Gaussian targets with occasional additive outliers.

    Args:
        n_train (int): Number of training rows.
        n_test (int): Number of test rows.
        seed (int): RNG seed.

    Returns:
        DatasetBundle: Dataset with rare additive spikes.

    Raises:
        None.

    Side Effects:
        None.

    Complexity:
        O(n_train + n_test).
    """

    def builder(
        rng: np.random.Generator,
        n: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = rng.normal(size=(n, 2))
        y = 0.6 * X[:, 0] - 0.2 * X[:, 1] + 0.2 * rng.normal(size=n)
        mask = rng.random(n) < OUTLIER_PROB
        y[mask] += rng.normal(loc=3.0, scale=0.5, size=mask.sum())
        return X, y

    rng = np.random.default_rng(seed)
    return _split(rng, builder, n_train, n_test)


def load_dataset(name: str) -> DatasetBundle:
    """Return a dataset bundle by name.

    Args:
        name (str): Dataset identifier (e.g. `"heteroscedastic"`).

    Returns:
        DatasetBundle: Train/test arrays for the requested distribution shift.

    Raises:
        ValueError: If the dataset name is unknown.

    Side Effects:
        None.

    Complexity:
        O(n) for the relevant dataset sampler.
    """
    if name == "heteroscedastic":
        return heteroscedastic()
    if name == "multimodal":
        return multimodal()
    if name == "heavy_tail":
        return heavy_tail()
    if name == "skewed":
        return skewed()
    if name == "outliers":
        return outliers()
    msg = f"Unknown benchmark dataset '{name}'."
    raise ValueError(msg)
