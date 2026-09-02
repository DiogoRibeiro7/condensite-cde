"""Lightweight baseline models used for benchmarking Condensite."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig


@dataclass(slots=True)
class TrainingConfig:
    """Common training hyperparameters for simple baselines."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    epochs: int = 6
    batch_size: int = 128
    lr: float = 2e-3


class FeedForward(nn.Module):
    """Plain MLP with ReLU activations for quick baseline experiments."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> None:
        """Build the sequential dense network.

        Args:
            input_dim (int): Number of input features.
            hidden_sizes (tuple[int, ...]): Width of each hidden layer.
            output_dim (int): Dimension of the final linear layer.

        Returns:
            None.

        Raises:
            None.

        Side Effects:
            Registers `torch.nn` modules stored on `self.net`.

        Complexity:
            O(len(hidden_sizes)) parameter creations.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            prev = size
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP forward pass.

        Args:
            x (torch.Tensor): Input batch `[batch, features]`.

        Returns:
            torch.Tensor: Output `[batch, output_dim]`.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(sum(layer_sizes)) FLOPs dominated by dense matmuls.
        """
        return self.net(x)


class BenchmarkModel:
    """Interface implemented by each benchmark baseline."""

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BenchmarkModel:
        """Train the model on the provided dataset.

        Args:
            X (NDArray[np.float64]): Training features.
            y (NDArray[np.float64]): Training targets.

        Returns:
            BenchmarkModel: Implementations should return `self`.

        Raises:
            NotImplementedError: Base class requires overrides.

        Side Effects:
            Implementation specific.

        Complexity:
            Implementation specific.
        """
        raise NotImplementedError

    def predict_pdf(
        self,
        X: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict pdf values on `y_grid`.

        Args:
            X (NDArray[np.float64]): Feature matrix.
            y_grid (NDArray[np.float64]): Grid to evaluate.

        Returns:
            NDArray[np.float64]: Pdf values `[n_samples, grid_size]`.

        Raises:
            NotImplementedError: Base class requires overrides.

        Side Effects:
            Implementation specific.

        Complexity:
            Implementation specific.
        """
        raise NotImplementedError

    def predict_cdf(
        self,
        X: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict cdf values evaluated on `y_grid`.

        Args:
            X (NDArray[np.float64]): Feature matrix.
            y_grid (NDArray[np.float64]): Grid to evaluate.

        Returns:
            NDArray[np.float64]: Cdf values `[n_samples, grid_size]`.

        Raises:
            NotImplementedError: Base class requires overrides.

        Side Effects:
            Implementation specific.

        Complexity:
            Implementation specific.
        """
        raise NotImplementedError


@dataclass
class CondensiteBaseline(BenchmarkModel):
    config: CondensiteTorchCDEConfig
    random_seed: int = 0

    def __post_init__(self) -> None:
        """Create a private copy of the estimator to avoid config mutation.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.

        Side Effects:
            Copies the configuration deeply.

        Complexity:
            O(size(config)).
        """
        self.estimator = CondensiteTorchCDE(
            config=copy.deepcopy(self.config),
            random_seed=self.random_seed,
        )

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BenchmarkModel:
        """Fit the Condensite model for use as a baseline.

        Args:
            X (NDArray[np.float64]): Training features.
            y (NDArray[np.float64]): Training targets.

        Returns:
            BenchmarkModel: The fitted baseline (`self`).

        Raises:
            ValueError: Propagated from Condensite if data is invalid.

        Side Effects:
            Trains the wrapped estimator in-place.

        Complexity:
            Matches the underlying Condensite training cost.
        """
        self.estimator.fit(X, y)
        return self

    def predict_pdf(
        self,
        X: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return Condensite pdf predictions.

        Args:
            X (NDArray[np.float64]): Feature matrix.
            y_grid (NDArray[np.float64]): Grid to evaluate.

        Returns:
            NDArray[np.float64]: Pdf values `[n_samples, grid_size]`.

        Raises:
            RuntimeError: If the estimator has not been fitted.

        Side Effects:
            None.

        Complexity:
            O(n_samples * grid_size).
        """
        return self.estimator.predict_density(X, y_grid)

    def predict_cdf(
        self,
        X: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return Condensite cdf predictions.

        Args:
            X (NDArray[np.float64]): Feature matrix.
            y_grid (NDArray[np.float64]): Grid to evaluate.

        Returns:
            NDArray[np.float64]: Cdf values `[n_samples, grid_size]`.

        Raises:
            RuntimeError: If the estimator has not been fitted.

        Side Effects:
            None.

        Complexity:
            O(n_samples * grid_size).
        """
        return self.estimator.predict_cdf(X, y_grid)


class ConditionalGaussianBaseline(BenchmarkModel):
    """Mean/variance network modeling Gaussian conditionals."""

    def __init__(
        self,
        input_dim: int,
        *,
        training: TrainingConfig | None = None,
    ) -> None:
        """Configure the Gaussian baseline.

        Args:
            input_dim (int): Number of input features.
            training (TrainingConfig | None): Overrides for network hyperparameters.

        Returns:
            None.

        Raises:
            None.

        Side Effects:
            Instantiates a CPU-only neural network.

        Complexity:
            O(sum(hidden_sizes)) parameter allocations.
        """
        cfg = training or TrainingConfig()
        self.model = FeedForward(input_dim, cfg.hidden_sizes, 2)
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size
        self.lr = cfg.lr
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BenchmarkModel:
        """Train mean and log-std heads via maximum likelihood.

        Args:
            X (NDArray[np.float64]): Training features.
            y (NDArray[np.float64]): Training targets.

        Returns:
            BenchmarkModel: The fitted baseline (`self`).

        Raises:
            None.

        Side Effects:
            Updates network parameters in-place.

        Complexity:
            O(epochs * n_samples * hidden_dim).
        """
        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        torch.manual_seed(0)
        for _ in range(self.epochs):
            for xb_batch, yb_batch in loader:
                xb = xb_batch.to(self.device)
                yb = yb_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                mean = preds[:, :1]
                log_std = preds[:, 1:2]
                std = torch.exp(log_std).clamp(min=1e-4)
                loss = log_std + 0.5 * ((yb - mean) / std) ** 2 + 0.5 * math.log(2 * math.pi)
                loss.mean().backward()
                optimizer.step()
        return self

    def _predict_params(
        self,
        X: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Infer mean/std for each row in `X`.

        Args:
            X (NDArray[np.float64]): Feature matrix.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: Means and stds for each row.

        Raises:
            None.

        Side Effects:
            Sets module to eval mode temporarily.

        Complexity:
            O(n_samples * hidden_dim).
        """
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
            preds = self.model(tensor)
        mean = preds[:, 0].cpu().numpy()
        std = torch.exp(preds[:, 1]).cpu().numpy()
        std = np.clip(std, 1e-3, None)
        return mean, std

    def predict_pdf(
        self,
        X: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate Gaussian pdfs on the requested grid.

        Args:
            X (NDArray[np.float64]): Feature matrix.
            y_grid (NDArray[np.float64]): Grid to evaluate.

        Returns:
            NDArray[np.float64]: Pdf values `[n_samples, grid_size]`.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(n_samples * grid_size).
        """
        mean, std = self._predict_params(X)
        diff = (y_grid[None, :] - mean[:, None]) / std[:, None]
        pdf = (1.0 / (std[:, None] * np.sqrt(2 * math.pi))) * np.exp(-0.5 * diff**2)
        return pdf

    def predict_cdf(
        self,
        X: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate Gaussian cdfs on the requested grid.

        Args:
            X (NDArray[np.float64]): Feature matrix.
            y_grid (NDArray[np.float64]): Grid to evaluate.

        Returns:
            NDArray[np.float64]: Cdf values `[n_samples, grid_size]`.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(n_samples * grid_size).
        """
        mean, std = self._predict_params(X)
        z = (y_grid[None, :] - mean[:, None]) / (std[:, None] * np.sqrt(2))
        cdf = 0.5 * (1.0 + torch.erf(torch.from_numpy(z))).numpy()
        return np.clip(cdf, 0.0, 1.0)


class QuantileRegressionBaseline(BenchmarkModel):
    """Independent quantile heads trained with pinball loss."""

    def __init__(
        self,
        input_dim: int,
        *,
        taus: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
        training: TrainingConfig | None = None,
    ) -> None:
        """Configure the quantile regression baseline.

        Args:
            input_dim (int): Feature dimension.
            taus (tuple[float, ...]): Quantiles to predict.
            training (TrainingConfig | None): Optional hyperparameter overrides.

        Returns:
            None.

        Raises:
            None.

        Side Effects:
            Instantiates a CPU network.

        Complexity:
            O(sum(hidden_sizes)) parameter creation.
        """
        cfg = training or TrainingConfig()
        self.taus = np.array(sorted(taus), dtype=np.float64)
        self.model = FeedForward(input_dim, cfg.hidden_sizes, len(self.taus))
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size
        self.lr = cfg.lr
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BenchmarkModel:
        """Train each quantile head with the asymmetric pinball loss.

        Args:
            X (NDArray[np.float64]): Training features.
            y (NDArray[np.float64]): Targets.

        Returns:
            BenchmarkModel: The fitted baseline (`self`).

        Raises:
            None.

        Side Effects:
            Updates network weights.

        Complexity:
            O(epochs * n_samples * hidden_dim).
        """
        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        tau_tensor = torch.from_numpy(self.taus.astype(np.float32)).to(self.device)
        torch.manual_seed(0)
        for _ in range(self.epochs):
            for xb_batch, yb_batch in loader:
                xb = xb_batch.to(self.device)
                yb = yb_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                diff = yb - preds
                losses = torch.where(diff >= 0, tau_tensor * diff, (tau_tensor - 1.0) * diff)
                losses.mean().backward()
                optimizer.step()
        return self

    def _predict_quantiles_base(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict and sort quantile values for each observation.

        Args:
            X (NDArray[np.float64]): Feature matrix.

        Returns:
            NDArray[np.float64]: Sorted quantile predictions.

        Raises:
            None.

        Side Effects:
            Sets module to eval mode temporarily.

        Complexity:
            O(n_samples * hidden_dim).
        """
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
            preds = self.model(tensor).cpu().numpy()
        preds.sort(axis=1)
        return preds

    def predict_cdf(
        self,
        X: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Interpolate the cdf based on predicted quantiles.

        Args:
            X (NDArray[np.float64]): Feature matrix.
            y_grid (NDArray[np.float64]): Grid to evaluate.

        Returns:
            NDArray[np.float64]: Cdf values `[n_samples, grid_size]`.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(n_samples * grid_size).
        """
        quantiles = self._predict_quantiles_base(X)
        cdf = np.empty((X.shape[0], y_grid.size), dtype=np.float64)
        for idx in range(X.shape[0]):
            cdf[idx] = np.interp(y_grid, quantiles[idx], self.taus, left=0.0, right=1.0)
        return np.clip(cdf, 0.0, 1.0)

    def predict_pdf(
        self,
        X: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Approximate the pdf by differentiating the interpolated cdf.

        Args:
            X (NDArray[np.float64]): Feature matrix.
            y_grid (NDArray[np.float64]): Grid to evaluate.

        Returns:
            NDArray[np.float64]: Pseudo-pdf values `[n_samples, grid_size]`.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(n_samples * grid_size).
        """
        cdf = self.predict_cdf(X, y_grid)
        diffs = np.diff(cdf, axis=1) / np.diff(y_grid)
        pdf = np.concatenate([diffs[:, :1], diffs], axis=1)
        return np.clip(pdf, 0.0, None)


class MixtureDensityNetworkBaseline(BenchmarkModel):
    """Mixture of Gaussians baseline estimated with MDN loss."""

    def __init__(
        self,
        input_dim: int,
        *,
        n_components: int = 3,
        training: TrainingConfig | None = None,
    ) -> None:
        """Configure the MDN baseline.

        Args:
            input_dim (int): Feature dimension.
            n_components (int): Number of mixture components.
            training (TrainingConfig | None): Optional hyperparameter overrides.

        Returns:
            None.

        Raises:
            None.

        Side Effects:
            Instantiates a CPU network.

        Complexity:
            O(sum(hidden_sizes)) parameter creation.
        """
        cfg = training or TrainingConfig()
        self.n_components = n_components
        output_dim = n_components * 3
        self.model = FeedForward(input_dim, cfg.hidden_sizes, output_dim)
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size
        self.lr = cfg.lr
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BenchmarkModel:
        """Optimize the MDN loss via maximum likelihood.

        Args:
            X (NDArray[np.float64]): Training features.
            y (NDArray[np.float64]): Targets.

        Returns:
            BenchmarkModel: The fitted baseline (`self`).

        Raises:
            None.

        Side Effects:
            Updates network weights.

        Complexity:
            O(epochs * n_samples * hidden_dim).
        """
        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        torch.manual_seed(0)
        for _ in range(self.epochs):
            for xb_batch, yb_batch in loader:
                xb = xb_batch.to(self.device)
                yb = yb_batch.to(self.device)
                optimizer.zero_grad()
                weights, means, std = self._forward_params(xb)
                diff = (yb - means) / std
                component_pdf = (1.0 / (std * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * diff**2)
                mix_pdf = torch.sum(weights * component_pdf, dim=1, keepdim=True)
                loss = -torch.log(mix_pdf + 1e-8).mean()
                loss.backward()
                optimizer.step()
        return self

    def _forward_params(self, xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split network outputs into weights/means/std parameters.

        Args:
            xb (torch.Tensor): Batch features on the correct device.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Weights, means, std tensors.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(batch * hidden_dim).
        """
        preds = self.model(xb)
        weights_raw, means, log_std = torch.chunk(preds, 3, dim=1)
        weights = torch.softmax(weights_raw.view(-1, self.n_components), dim=1)
        means = means.view(-1, self.n_components)
        std = torch.exp(log_std.view(-1, self.n_components)).clamp(min=1e-3)
        return weights, means, std

    def _predict_params(
        self,
        X: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Infer MDN parameters for each input row.

        Args:
            X (NDArray[np.float64]): Feature matrix.

        Returns:
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
            ]: weights/means/std arrays.

        Raises:
            None.

        Side Effects:
            Temporarily switches to eval mode.

        Complexity:
            O(n_samples * hidden_dim).
        """
        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X.astype(np.float32)).to(self.device)
            weights, means, std = self._forward_params(xb)
        return weights.cpu().numpy(), means.cpu().numpy(), std.cpu().numpy()

    def predict_pdf(
        self,
        X: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate the MDN pdf across the provided grid.

        Args:
            X (NDArray[np.float64]): Feature matrix.
            y_grid (NDArray[np.float64]): Grid to evaluate.

        Returns:
            NDArray[np.float64]: Pdf values `[n_samples, grid_size]`.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(n_samples * grid_size * n_components).
        """
        weights, means, std = self._predict_params(X)
        diff = (y_grid[None, None, :] - means[:, :, None]) / std[:, :, None]
        component_pdf = (1.0 / (std[:, :, None] * np.sqrt(2 * math.pi))) * np.exp(-0.5 * diff**2)
        pdf = np.sum(weights[:, :, None] * component_pdf, axis=1)
        return pdf

    def predict_cdf(
        self,
        X: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate the MDN cdf across the provided grid.

        Args:
            X (NDArray[np.float64]): Feature matrix.
            y_grid (NDArray[np.float64]): Grid to evaluate.

        Returns:
            NDArray[np.float64]: Cdf values `[n_samples, grid_size]`.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(n_samples * grid_size * n_components).
        """
        weights, means, std = self._predict_params(X)
        z = (y_grid[None, None, :] - means[:, :, None]) / (std[:, :, None] * np.sqrt(2))
        component_cdf = 0.5 * (1.0 + torch.erf(torch.from_numpy(z))).numpy()
        cdf = np.sum(weights[:, :, None] * component_cdf, axis=1)
        return np.clip(cdf, 0.0, 1.0)
