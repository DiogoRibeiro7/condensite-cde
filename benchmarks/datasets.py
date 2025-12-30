from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class DatasetBundle:
    def __init__(self, X_train: NDArray[np.float64], y_train: NDArray[np.float64], X_test: NDArray[np.float64], y_test: NDArray[np.float64]) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


def _split(generator: np.random.Generator, builder, n_train: int, n_test: int) -> DatasetBundle:
    X_train, y_train = builder(generator, n_train)
    X_test, y_test = builder(generator, n_test)
    return DatasetBundle(X_train, y_train, X_test, y_test)


def heteroscedastic(n_train: int = 400, n_test: int = 200, seed: int = 0) -> DatasetBundle:
    def builder(rng: np.random.Generator, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = rng.uniform(-2.0, 2.0, size=(n, 2))
        noise_scale = 0.1 + 0.35 * (np.sin(X[:, 0]) ** 2)
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + noise_scale * rng.normal(size=n)
        return X, y

    rng = np.random.default_rng(seed)
    return _split(rng, builder, n_train, n_test)


def multimodal(n_train: int = 400, n_test: int = 200, seed: int = 1) -> DatasetBundle:
    def builder(rng: np.random.Generator, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = rng.normal(size=(n, 2))
        mode_choice = rng.integers(0, 2, size=n)
        base = np.sin(2 * X[:, 0]) + 0.25 * X[:, 1]
        y = np.where(mode_choice == 0, base + 0.5, -base - 0.5)
        y += 0.2 * rng.normal(size=n)
        return X, y

    rng = np.random.default_rng(seed)
    return _split(rng, builder, n_train, n_test)


def heavy_tail(n_train: int = 400, n_test: int = 200, seed: int = 2) -> DatasetBundle:
    def builder(rng: np.random.Generator, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = rng.normal(size=(n, 2))
        noise = 0.2 * rng.standard_t(df=3, size=n)
        y = 0.8 * X[:, 0] - 0.3 * X[:, 1] + noise
        return X, y

    rng = np.random.default_rng(seed)
    return _split(rng, builder, n_train, n_test)


def skewed(n_train: int = 400, n_test: int = 200, seed: int = 3) -> DatasetBundle:
    def builder(rng: np.random.Generator, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = rng.normal(size=(n, 2))
        skew_noise = rng.lognormal(mean=0.0, sigma=0.3, size=n) - 1.0
        y = 0.5 * np.sin(X[:, 0]) + 0.3 * X[:, 1] + 0.2 * skew_noise
        return X, y

    rng = np.random.default_rng(seed)
    return _split(rng, builder, n_train, n_test)


def outliers(n_train: int = 400, n_test: int = 200, seed: int = 4) -> DatasetBundle:
    def builder(rng: np.random.Generator, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = rng.normal(size=(n, 2))
        y = 0.6 * X[:, 0] - 0.2 * X[:, 1] + 0.2 * rng.normal(size=n)
        mask = rng.random(n) < 0.05
        y[mask] += rng.normal(loc=3.0, scale=0.5, size=mask.sum())
        return X, y

    rng = np.random.default_rng(seed)
    return _split(rng, builder, n_train, n_test)


def load_dataset(name: str) -> DatasetBundle:
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
