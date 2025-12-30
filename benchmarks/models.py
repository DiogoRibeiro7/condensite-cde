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


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            prev = size
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class BenchmarkModel:
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BenchmarkModel:
        raise NotImplementedError

    def predict_pdf(self, X: NDArray[np.float64], y_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def predict_cdf(self, X: NDArray[np.float64], y_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError


@dataclass
class CondensiteBaseline(BenchmarkModel):
    config: CondensiteTorchCDEConfig
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.estimator = CondensiteTorchCDE(config=copy.deepcopy(self.config), random_seed=self.random_seed)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BenchmarkModel:
        self.estimator.fit(X, y)
        return self

    def predict_pdf(self, X: NDArray[np.float64], y_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.estimator.predict_density(X, y_grid)

    def predict_cdf(self, X: NDArray[np.float64], y_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.estimator.predict_cdf(X, y_grid)


class ConditionalGaussianBaseline(BenchmarkModel):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = (64, 64), epochs: int = 6, batch_size: int = 128, lr: float = 2e-3) -> None:
        self.model = FeedForward(input_dim, hidden_sizes, 2)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BenchmarkModel:
        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        torch.manual_seed(0)
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                mean = preds[:, :1]
                log_std = preds[:, 1:2]
                std = torch.exp(log_std).clamp(min=1e-4)
                loss = log_std + 0.5 * ((yb - mean) / std) ** 2 + 0.5 * math.log(2 * math.pi)
                loss.mean().backward()
                optimizer.step()
        return self

    def _predict_params(self, X: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
            preds = self.model(tensor)
        mean = preds[:, 0].cpu().numpy()
        std = torch.exp(preds[:, 1]).cpu().numpy()
        std = np.clip(std, 1e-3, None)
        return mean, std

    def predict_pdf(self, X: NDArray[np.float64], y_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        mean, std = self._predict_params(X)
        diff = (y_grid[None, :] - mean[:, None]) / std[:, None]
        pdf = (1.0 / (std[:, None] * np.sqrt(2 * math.pi))) * np.exp(-0.5 * diff**2)
        return pdf

    def predict_cdf(self, X: NDArray[np.float64], y_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        mean, std = self._predict_params(X)
        z = (y_grid[None, :] - mean[:, None]) / (std[:, None] * np.sqrt(2))
        cdf = 0.5 * (1.0 + torch.erf(torch.from_numpy(z))).numpy()
        return np.clip(cdf, 0.0, 1.0)


class QuantileRegressionBaseline(BenchmarkModel):
    def __init__(self, input_dim: int, taus: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95), hidden_sizes: tuple[int, ...] = (64, 64), epochs: int = 6, batch_size: int = 128, lr: float = 2e-3) -> None:
        self.taus = np.array(sorted(taus), dtype=np.float64)
        self.model = FeedForward(input_dim, hidden_sizes, len(self.taus))
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BenchmarkModel:
        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        tau_tensor = torch.from_numpy(self.taus.astype(np.float32)).to(self.device)
        torch.manual_seed(0)
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                diff = yb - preds
                losses = torch.where(diff >= 0, tau_tensor * diff, (tau_tensor - 1.0) * diff)
                losses.mean().backward()
                optimizer.step()
        return self

    def _predict_quantiles_base(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
            preds = self.model(tensor).cpu().numpy()
        preds.sort(axis=1)
        return preds

    def predict_cdf(self, X: NDArray[np.float64], y_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        quantiles = self._predict_quantiles_base(X)
        cdf = np.empty((X.shape[0], y_grid.size), dtype=np.float64)
        for idx in range(X.shape[0]):
            cdf[idx] = np.interp(y_grid, quantiles[idx], self.taus, left=0.0, right=1.0)
        return np.clip(cdf, 0.0, 1.0)

    def predict_pdf(self, X: NDArray[np.float64], y_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        cdf = self.predict_cdf(X, y_grid)
        diffs = np.diff(cdf, axis=1) / np.diff(y_grid)
        pdf = np.concatenate([diffs[:, :1], diffs], axis=1)
        return np.clip(pdf, 0.0, None)


class MixtureDensityNetworkBaseline(BenchmarkModel):
    def __init__(self, input_dim: int, n_components: int = 3, hidden_sizes: tuple[int, ...] = (64, 64), epochs: int = 6, batch_size: int = 128, lr: float = 2e-3) -> None:
        self.n_components = n_components
        output_dim = n_components * 3
        self.model = FeedForward(input_dim, hidden_sizes, output_dim)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BenchmarkModel:
        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        torch.manual_seed(0)
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
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
        preds = self.model(xb)
        weights_raw, means, log_std = torch.chunk(preds, 3, dim=1)
        weights = torch.softmax(weights_raw.view(-1, self.n_components), dim=1)
        means = means.view(-1, self.n_components)
        std = torch.exp(log_std.view(-1, self.n_components)).clamp(min=1e-3)
        return weights, means, std

    def _predict_params(self, X: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X.astype(np.float32)).to(self.device)
            weights, means, std = self._forward_params(xb)
        return weights.cpu().numpy(), means.cpu().numpy(), std.cpu().numpy()

    def predict_pdf(self, X: NDArray[np.float64], y_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        weights, means, std = self._predict_params(X)
        diff = (y_grid[None, None, :] - means[:, :, None]) / std[:, :, None]
        component_pdf = (1.0 / (std[:, :, None] * np.sqrt(2 * math.pi))) * np.exp(-0.5 * diff**2)
        pdf = np.sum(weights[:, :, None] * component_pdf, axis=1)
        return pdf

    def predict_cdf(self, X: NDArray[np.float64], y_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        weights, means, std = self._predict_params(X)
        z = (y_grid[None, None, :] - means[:, :, None]) / (std[:, :, None] * np.sqrt(2))
        component_cdf = 0.5 * (1.0 + torch.erf(torch.from_numpy(z))).numpy()
        cdf = np.sum(weights[:, :, None] * component_cdf, axis=1)
        return np.clip(cdf, 0.0, 1.0)
