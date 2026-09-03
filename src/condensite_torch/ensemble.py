"""Deep ensemble wrapper for CondensiteTorchCDE."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .estimator import CondensiteTorchCDE, CondensiteTorchCDEConfig


class EnsembleCondensite:
    """Train multiple CondensiteTorchCDE models and aggregate predictions."""

    def __init__(
        self,
        base_config: CondensiteTorchCDEConfig,
        *,
        n_models: int = 3,
        bootstrap: bool = False,
        random_seed: int = 0,
    ) -> None:
        """Configure the ensemble wrapper.

        Args:
            base_config (CondensiteTorchCDEConfig): Shared configuration for each member.
            n_models (int): Number of members to train.
            bootstrap (bool): Whether to resample rows with replacement per member.
            random_seed (int): Seed for bootstrap sampling and member seeds.

        Returns:
            None.

        Raises:
            ValueError: If `n_models <= 0`.

        Side Effects:
            Duplicates `base_config` to decouple member mutations.
        """
        if n_models <= 0:
            msg = "n_models must be positive."
            raise ValueError(msg)
        self.base_config = copy.deepcopy(base_config)
        self.n_models = int(n_models)
        self.bootstrap = bool(bootstrap)
        self.random_seed = int(random_seed)
        self._members: list[CondensiteTorchCDE] = []
        self._fitted = False

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> EnsembleCondensite:
        """Train each ensemble member sequentially.

        Args:
            X (NDArray[np.floating]): Feature matrix.
            y (NDArray[np.floating]): Target vector.

        Returns:
            EnsembleCondensite: The fitted ensemble.

        Raises:
            ValueError: If `X` and `y` have mismatched rows.

        Side Effects:
            Trains `n_models` Condensite estimators in-place.

        Complexity:
            O(n_models * base_training_cost).
        """
        X_arr = np.asarray(X, dtype=object)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if X_arr.shape[0] != y_arr.shape[0]:
            msg = "X and y must have the same number of rows."
            raise ValueError(msg)
        self._members.clear()
        rng = np.random.default_rng(self.random_seed)
        for idx in range(self.n_models):
            config = copy.deepcopy(self.base_config)
            if config.epistemic_mode == "ensemble":
                config.epistemic_mode = "none"
            estimator = CondensiteTorchCDE(config=config, random_seed=self.random_seed + idx)
            if self.bootstrap:
                indices = rng.integers(0, X_arr.shape[0], size=X_arr.shape[0])
                X_train = X_arr[indices]
                y_train = y_arr[indices]
            else:
                X_train = X_arr
                y_train = y_arr
            estimator.fit(X_train, y_train)
            self._members.append(estimator)
        self._fitted = True
        return self

    def predict_density(
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating],
        *,
        head: int | str | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return mean/variance of the ensemble pdf predictions.

        Args:
            X (NDArray[np.floating]): Feature matrix.
            y_grid (NDArray[np.floating]): Grid to evaluate density on.
            head (int | str | None): Optional bandwidth head selection forwarded to members.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: `(mean, variance)` across
                ensemble members.

        Raises:
            RuntimeError: If `fit` has not been executed.

        Side Effects:
            None.

        Complexity:
            O(n_models * n_samples * grid_size).
        """
        outputs = self._collect(lambda model: model.predict_density(X, y_grid, head=head))
        return _aggregate(outputs)

    def predict_cdf(
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating],
        *,
        head: int | str | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return mean/variance of the ensemble cdf predictions.

        Args:
            X (NDArray[np.floating]): Feature matrix.
            y_grid (NDArray[np.floating]): Grid to evaluate the cdf on.
            head (int | str | None): Optional bandwidth head forwarded to members.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: `(mean, variance)` along ensemble axis.

        Raises:
            RuntimeError: If the ensemble is not fitted.

        Side Effects:
            None.

        Complexity:
            O(n_models * n_samples * grid_size).
        """
        outputs = self._collect(lambda model: model.predict_cdf(X, y_grid, head=head))
        return _aggregate(outputs)

    def predict_quantile(
        self,
        X: NDArray[np.floating],
        q: NDArray[np.floating] | float,
        *,
        y_grid: NDArray[np.floating] | None = None,
        head: int | str | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return mean/variance of ensemble quantile predictions.

        Args:
            X (NDArray[np.floating]): Feature matrix.
            q (NDArray[np.floating] | float): Quantile level(s) to evaluate.
            y_grid (NDArray[np.floating] | None): Optional grid for inversion.
            head (int | str | None): Bandwidth head forwarded to members.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: `(mean, variance)` along ensemble axis.

        Raises:
            RuntimeError: If ensemble is not fitted.

        Side Effects:
            None.

        Complexity:
            O(n_models * n_samples * grid_size).
        """
        outputs = self._collect(
            lambda model: model.predict_quantile(X, q, y_grid=y_grid, head=head),
        )
        return _aggregate(outputs)

    def save(self, path: str | Path) -> None:
        """Persist every ensemble member to disk alongside metadata.

        Args:
            path (str | Path): Destination directory.

        Returns:
            None.

        Raises:
            RuntimeError: If the ensemble has not been fitted.

        Side Effects:
            Writes metadata + per-member artifacts to disk.

        Complexity:
            O(n_models * params) serialization.
        """
        self._ensure_fitted()
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)
        metadata = {
            "n_models": self.n_models,
            "bootstrap": self.bootstrap,
            "random_seed": self.random_seed,
        }
        (base / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for idx, member in enumerate(self._members):
            member.save(base / f"member_{idx}")

    @classmethod
    def load(cls, path: str | Path) -> EnsembleCondensite:
        """Restore an ensemble from disk.

        Args:
            path (str | Path): Directory created by `save`.

        Returns:
            EnsembleCondensite: Restored ensemble with fitted members.

        Raises:
            FileNotFoundError: If metadata or member folders are missing.

        Side Effects:
            Loads models from disk using CPU map location.

        Complexity:
            O(n_models * params).
        """
        base = Path(path)
        data = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
        dummy_config = CondensiteTorchCDEConfig()
        ensemble = cls(
            base_config=dummy_config,
            n_models=int(data["n_models"]),
            bootstrap=bool(data["bootstrap"]),
            random_seed=int(data["random_seed"]),
        )
        ensemble._members = [
            CondensiteTorchCDE.load(base / f"member_{idx}", map_location="cpu")
            for idx in range(ensemble.n_models)
        ]
        ensemble._fitted = True
        return ensemble

    def _collect(
        self,
        fn: Callable[[CondensiteTorchCDE], NDArray[np.float64]],
    ) -> Sequence[NDArray[np.float64]]:
        """Apply `fn` to every fitted member and return the stacked outputs.

        Args:
            fn (Callable): Callable applied to each member.

        Returns:
            Sequence[NDArray[np.float64]]: Per-member outputs.

        Raises:
            RuntimeError: If ensemble is not fitted.

        Side Effects:
            None.

        Complexity:
            O(n_models) calls to `fn`.
        """
        self._ensure_fitted()
        return [fn(member) for member in self._members]

    def _ensure_fitted(self) -> None:
        """Guard methods that require trained ensemble members.

        Raises:
            RuntimeError: If `fit` hasn't executed yet.
        """
        if not self._fitted or not self._members:
            msg = "Call fit() before requesting predictions."
            raise RuntimeError(msg)


def _aggregate(
    samples: Sequence[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate samples by computing mean/variance along the ensemble axis.

    Args:
        samples (Sequence[NDArray[np.float64]]): List of per-member predictions.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Mean and variance.

    Raises:
        ValueError: If no samples are provided.

    Side Effects:
        None.

    Complexity:
        O(n_models * n_samples * grid_size).
    """
    if not samples:
        msg = "samples must contain at least one array."
        raise ValueError(msg)
    stack = np.stack(samples, axis=0).astype(np.float64)
    mean = np.mean(stack, axis=0)
    var = np.var(stack, axis=0, ddof=0)
    return mean, var


__all__ = ("EnsembleCondensite",)
