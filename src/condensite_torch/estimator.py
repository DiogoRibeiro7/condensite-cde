"""Condensité estimator with streamed auxiliary sampling and torch training."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .aux_sampling import sample_yprime
from .kernels import kernel_h_torch
from .models import ActivationFactory, MLPRegressor, MLPRegressorConfig
from .scalers import MinMaxScaler1D, StandardScaler


@dataclass
class CondensiteTorchCDEConfig:
    """Configuration describing model, optimizer, and training behaviour."""

    hidden_sizes: Sequence[int] = (128, 128)
    activation: ActivationFactory | nn.Module | None = None
    dropout: float = 0.0
    positive_output: bool = True
    bandwidth: float = 0.1
    m_aux: int = 32
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    sampler: str = "stratified"
    amp: bool = False
    val_fraction: float = 0.1
    device: str | torch.device = "cpu"


class CondensiteTorchCDE:
    """Streamed y' conditional density estimator with Sobol/stratified sampling."""

    def __init__(
        self,
        config: CondensiteTorchCDEConfig | None = None,
        random_seed: int = 0,
    ) -> None:
        self.config = config or CondensiteTorchCDEConfig()
        self.random_seed = int(random_seed)
        self._rng = np.random.default_rng(self.random_seed)
        self._sampler_call = 0
        self._device = torch.device(self.config.device)
        self.x_scaler: StandardScaler | None = None
        self.y_scaler: MinMaxScaler1D | None = None
        self.model: MLPRegressor | None = None
        self.training_history: list[dict[str, float]] = []
        self._fitted = False

    # ------------------------------------------------------------------ Training
    def fit(  # noqa: PLR0914, PLR0915
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        *,
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> CondensiteTorchCDE:
        X_arr: NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        y_arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64).reshape(-1)
        expected_dim = 2
        if X_arr.ndim != expected_dim:
            msg = f"Expected 2D array for X, received {X_arr.shape}"
            raise ValueError(msg)
        if X_arr.shape[0] != y_arr.shape[0]:
            msg = "X and y must contain the same number of samples."
            raise ValueError(msg)

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.x_scaler = StandardScaler().fit(X_arr)
        self.y_scaler = MinMaxScaler1D().fit(y_arr)

        X_scaled: NDArray[np.float32] = self.x_scaler.transform(X_arr)
        y_scaled: NDArray[np.float32] = self.y_scaler.transform(y_arr)

        X_val_scaled: NDArray[np.float32] | None
        y_val_scaled: NDArray[np.float32] | None
        if X_val is None or y_val is None:
            X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled = self._train_val_split(
                X_scaled,
                y_scaled,
            )
        else:
            X_val_arr = np.asarray(X_val, dtype=np.float64)
            y_val_arr = np.asarray(y_val, dtype=np.float64).reshape(-1)
            X_val_scaled = self.x_scaler.transform(X_val_arr)
            y_val_scaled = self.y_scaler.transform(y_val_arr)
            X_train_scaled, y_train_scaled = X_scaled, y_scaled

        train_loader = self._build_loader(X_train_scaled, y_train_scaled, shuffle=True)
        val_loader = None
        if X_val_scaled is not None and y_val_scaled is not None:
            val_loader = self._build_loader(X_val_scaled, y_val_scaled, shuffle=False)

        input_dim = X_arr.shape[1] + 1
        model_config = MLPRegressorConfig(
            input_dim=input_dim,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            dropout=self.config.dropout,
            positive_output=self.config.positive_output,
        )
        self.model = MLPRegressor(model_config).to(self._device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        amp_enabled = self.config.amp and self._device.type == "cuda"
        scaler = GradScaler(enabled=amp_enabled)
        best_state: dict[str, Tensor] | None = None
        best_loss = math.inf
        epochs_without_improvement = 0
        self.training_history.clear()

        for epoch in range(self.config.epochs):
            train_loss = self._train_one_epoch(train_loader, optimizer, scaler, amp_enabled)
            val_loss = (
                self._evaluate(val_loader, amp_enabled) if val_loader is not None else train_loss
            )
            self.training_history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss},
            )

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._fitted = True
        return self

    def _train_val_split(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.float32],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32] | None,
        NDArray[np.float32] | None,
    ]:
        if self.config.val_fraction <= 0.0 or X.shape[0] <= 1:
            return X, y, None, None
        n_samples = X.shape[0]
        n_val = max(1, int(n_samples * self.config.val_fraction))
        if n_val >= n_samples:
            n_val = n_samples - 1
        indices = np.arange(n_samples)
        self._rng.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _build_loader(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.float32],
        *,
        shuffle: bool,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        tensor_x = torch.from_numpy(X)
        tensor_y = torch.from_numpy(y)
        dataset: TensorDataset = TensorDataset(tensor_x, tensor_y)
        dataset_typed = cast(Dataset[tuple[Tensor, Tensor]], dataset)
        return DataLoader(
            dataset_typed,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _train_one_epoch(
        self,
        loader: DataLoader[tuple[Tensor, Tensor]],
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        amp_enabled: bool,
    ) -> float:
        assert self.model is not None
        self.model.train()
        total_loss = 0.0
        batches = 0
        for _batch_idx, (X_batch, y_batch) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            features, targets = self._prepare_training_batch(
                X_batch.to(self._device),
                y_batch.to(self._device),
            )
            amp_context = torch.autocast("cuda") if amp_enabled else nullcontext()
            with amp_context:
                predictions = self.model(features)
                loss = torch.nn.functional.mse_loss(predictions, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            batches += 1
        return total_loss / max(batches, 1)

    def _prepare_training_batch(self, X_batch: Tensor, y_batch: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        batch_size = X_batch.shape[0]
        m_aux = cfg.m_aux
        sampler_seed = self._next_sampler_seed()
        y_prime = sample_yprime(
            cfg.sampler,
            (batch_size, m_aux),
            seed=sampler_seed,
            device=self._device,
            value_range=(0.0, 1.0),
        )
        kernels = kernel_h_torch(y_batch.unsqueeze(1), y_prime, cfg.bandwidth)
        features = torch.cat(
            [X_batch.unsqueeze(1).expand(-1, m_aux, -1), y_prime.unsqueeze(-1)],
            dim=-1,
        ).reshape(-1, X_batch.shape[1] + 1)
        targets = kernels.reshape(-1, 1)
        return features, targets

    def _next_sampler_seed(self) -> int:
        seed = self.random_seed + self._sampler_call
        self._sampler_call += 1
        return seed

    def _evaluate(
        self,
        loader: DataLoader[tuple[Tensor, Tensor]] | None,
        amp_enabled: bool,
    ) -> float:
        if loader is None:
            return math.inf
        assert self.model is not None
        self.model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                features, targets = self._prepare_training_batch(
                    X_batch.to(self._device),
                    y_batch.to(self._device),
                )
                amp_context = torch.autocast("cuda") if amp_enabled else nullcontext()
                with amp_context:
                    preds = self.model(features)
                    loss = torch.nn.functional.l1_loss(preds, targets)
                total_loss += float(loss.cpu())
                batches += 1
        return total_loss / max(batches, 1)

    # ------------------------------------------------------------------ Inference
    def predict_density(
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        self._ensure_fitted()
        assert self.x_scaler is not None
        assert self.y_scaler is not None
        X_arr: NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        y_grid_arr = self._validate_y_grid(y_grid)
        X_scaled = self.x_scaler.transform(X_arr)
        y_scaled = self.y_scaler.transform(y_grid_arr)
        x_tensor = torch.from_numpy(X_scaled).to(self._device)
        y_tensor = torch.from_numpy(y_scaled).to(self._device)
        grid_size = y_tensor.shape[0]

        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            x_expanded = x_tensor.unsqueeze(1).expand(-1, grid_size, -1)
            y_expanded = y_tensor.unsqueeze(0).expand(x_tensor.shape[0], -1).unsqueeze(-1)
            features = torch.cat([x_expanded, y_expanded], dim=-1).reshape(
                -1,
                x_tensor.shape[1] + 1,
            )
            preds = self.model(features).reshape(x_tensor.shape[0], grid_size)
        pdf_scaled = preds.cpu().numpy().astype(np.float64, copy=False)
        scale = float(self.y_scaler.data_range_)
        density: NDArray[np.float64] = pdf_scaled / scale
        return density

    def predict_cdf(
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        pdf = self.predict_density(X, y_grid)
        y_grid_arr = self._validate_y_grid(y_grid)
        diffs = np.diff(y_grid_arr)
        trapezoids = 0.5 * (pdf[:, 1:] + pdf[:, :-1]) * diffs
        cdf = np.concatenate(
            [np.zeros((pdf.shape[0], 1)), np.cumsum(trapezoids, axis=1)],
            axis=1,
        )
        cdf = np.maximum.accumulate(cdf, axis=1)
        norm = np.clip(cdf[:, -1:], 1e-8, None)
        cdf = np.clip(cdf / norm, 0.0, 1.0)
        cdf[:, -1] = 1.0
        result: NDArray[np.float64] = cdf.astype(np.float64, copy=False)
        return result

    def sample(
        self,
        X: NDArray[np.floating],
        n_samples: int,
        *,
        y_grid: NDArray[np.floating] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        if n_samples <= 0:
            msg = "n_samples must be positive."
            raise ValueError(msg)
        grid_arr = (
            self._validate_y_grid(y_grid) if y_grid is not None else self._default_y_grid()
        )
        cdf = self.predict_cdf(X, grid_arr)
        rng = np.random.default_rng(self.random_seed if seed is None else seed)
        samples = np.empty((X.shape[0], n_samples), dtype=np.float64)
        for idx, cdf_row in enumerate(cdf):
            u = rng.random(n_samples)
            samples[idx] = np.interp(u, cdf_row, grid_arr)
        return samples

    def _default_y_grid(self, num_points: int = 128) -> NDArray[np.float64]:
        self._ensure_fitted()
        assert self.y_scaler is not None
        return np.linspace(
            self.y_scaler.min_,
            self.y_scaler.max_,
            num_points,
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ Utilities
    def _ensure_fitted(self) -> None:
        if not self._fitted or self.model is None or self.x_scaler is None or self.y_scaler is None:
            msg = "Call fit() before requesting predictions."
            raise RuntimeError(msg)

    @staticmethod
    def _validate_y_grid(y_grid: NDArray[np.floating]) -> NDArray[np.float64]:
        arr: NDArray[np.float64] = np.asarray(y_grid, dtype=np.float64).reshape(-1)
        min_grid_points = 2
        if arr.ndim != 1 or arr.size < min_grid_points:
            msg = "y_grid must be a 1-D array with at least two points."
            raise ValueError(msg)
        if not np.all(np.diff(arr) > 0):
            msg = "y_grid must be strictly increasing."
            raise ValueError(msg)
        return arr


__all__: tuple[str, ...] = ("CondensiteTorchCDE", "CondensiteTorchCDEConfig")
