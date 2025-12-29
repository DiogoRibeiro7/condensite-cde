"""Condensité estimator with streamed auxiliary sampling and torch training."""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from condensite_cde.grids import make_y_grid

from .aux_sampling import ImportanceSampler, sample_yprime
from .kernels import kernel_h_torch
from .metrics import crps_from_cdf, nll_from_pdf
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
    bandwidths: Sequence[float] | None = None
    bandwidth_strategy: Literal["first", "mean", "best"] = "first"
    adaptive_bandwidth: Literal["none", "x"] = "none"
    normalization_lambda: float = 0.0
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
    monitor_metric: Literal["val_crps", "val_nll"] = "val_crps"


_MODEL_FILENAME = "model.pt"
_CONFIG_FILENAME = "config.json"
_SCALERS_FILENAME = "scalers.json"
_METADATA_FILENAME = "metadata.json"
_BANDWIDTH_MODEL_FILENAME = "bandwidth_net.pt"
_Y_REFERENCE_POINTS = 512
_ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "leakyrelu": nn.LeakyReLU,
}


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
        self._bandwidths: list[float] = []
        self._set_bandwidths()
        self.bandwidth_net: MLPRegressor | None = None
        self.x_scaler: StandardScaler | None = None
        self.y_scaler: MinMaxScaler1D | None = None
        self.model: MLPRegressor | None = None
        self.training_history: list[dict[str, float]] = []
        self._fitted = False
        self._y_train: NDArray[np.float64] | None = None
        self._best_head_index = 0
        self._importance_sampler: ImportanceSampler | None = None

    # ------------------------------------------------------------------ Training
    def fit(  # noqa: PLR0912, PLR0914, PLR0915
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

        self._set_bandwidths()
        self._y_train = y_arr.copy()

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.x_scaler = StandardScaler().fit(X_arr)
        self.y_scaler = MinMaxScaler1D().fit(y_arr)

        X_scaled: NDArray[np.float32] = self.x_scaler.transform(X_arr)
        y_scaled: NDArray[np.float32] = self.y_scaler.transform(y_arr)

        val_eval_data: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        if X_val is None or y_val is None:
            X_train_scaled, y_train_scaled, val_idx = self._train_val_split(X_scaled, y_scaled)
            if val_idx is not None and val_idx.size > 0:
                val_eval_data = (X_arr[val_idx], y_arr[val_idx])
        else:
            X_val_arr = np.asarray(X_val, dtype=np.float64)
            y_val_arr = np.asarray(y_val, dtype=np.float64).reshape(-1)
            if X_val_arr.shape[0] != y_val_arr.shape[0]:
                msg = "Validation X and y must match in the first dimension."
                raise ValueError(msg)
            self.x_scaler.transform(X_val_arr)  # Validate shape
            self.y_scaler.transform(y_val_arr)
            X_train_scaled, y_train_scaled = X_scaled, y_scaled
            if X_val_arr.size > 0:
                val_eval_data = (X_val_arr, y_val_arr)
            else:
                val_eval_data = None

        train_loader = self._build_loader(X_train_scaled, y_train_scaled, shuffle=True)
        self._configure_importance_sampler(y_train_scaled)

        x_dim = X_arr.shape[1]
        if self.config.adaptive_bandwidth == "x":
            self.bandwidth_net = self._build_bandwidth_net(x_dim)
        else:
            self.bandwidth_net = None

        input_dim = x_dim + 1
        model_config = MLPRegressorConfig(
            input_dim=input_dim,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            dropout=self.config.dropout,
            positive_output=self.config.positive_output,
            output_dim=len(self._bandwidths),
        )
        self.model = MLPRegressor(model_config).to(self._device)
        params = list(self.model.parameters())
        if self.bandwidth_net is not None:
            params.extend(self.bandwidth_net.parameters())
        optimizer = torch.optim.Adam(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        amp_enabled = self.config.amp and self._device.type == "cuda"
        scaler = GradScaler(enabled=amp_enabled)
        best_state: dict[str, Tensor] | None = None
        best_metric = math.inf
        epochs_without_improvement = 0
        self.training_history.clear()
        monitor_metric = self.config.monitor_metric
        valid_monitors = {"val_crps", "val_nll"}
        if monitor_metric not in valid_monitors:
            msg = f"monitor_metric must be one of {sorted(valid_monitors)}, got {monitor_metric!r}"
            raise ValueError(msg)
        val_grid: NDArray[np.float64] | None = None

        for epoch in range(self.config.epochs):
            train_loss = self._train_one_epoch(train_loader, optimizer, scaler, amp_enabled)
            record = {"epoch": epoch, "train_loss": train_loss}
            monitor_value = train_loss
            if val_eval_data is not None:
                if val_grid is None:
                    val_grid = self._default_y_grid()
                X_val_eval, y_val_eval = val_eval_data
                pdf_val = self._predict_density_internal(X_val_eval, val_grid)
                cdf_val = self._cdf_from_pdf(pdf_val, val_grid)
                val_crps = crps_from_cdf(y_val_eval, val_grid, cdf_val)
                val_nll = nll_from_pdf(y_val_eval, val_grid, pdf_val)
                record["val_crps"] = val_crps
                record["val_nll"] = val_nll
                monitor_value = record[monitor_metric]
            self.training_history.append(record)

            if monitor_value < best_metric - 1e-6:
                best_metric = monitor_value
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if val_eval_data is not None:
            eval_grid = val_grid if val_grid is not None else self._default_y_grid()
        else:
            eval_grid = None
        self._determine_best_head(val_eval_data, eval_grid)
        self._fitted = True
        return self

    def _train_val_split(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64] | None]:
        if self.config.val_fraction <= 0.0 or X.shape[0] <= 1:
            return X, y, None
        n_samples = X.shape[0]
        n_val = max(1, int(n_samples * self.config.val_fraction))
        if n_val >= n_samples:
            n_val = n_samples - 1
        indices = np.arange(n_samples)
        self._rng.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return X[train_idx], y[train_idx], val_idx

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
        if self.bandwidth_net is not None:
            self.bandwidth_net.train()
        total_loss = 0.0
        batches = 0
        for _batch_idx, (X_batch, y_batch) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            features, targets, weights = self._prepare_training_batch(
                X_batch.to(self._device),
                y_batch.to(self._device),
            )
            amp_context = torch.autocast("cuda") if amp_enabled else nullcontext()
            with amp_context:
                predictions = self.model(features)
                loss = self._weighted_mse_loss(predictions, targets, weights)
                if self.config.normalization_lambda > 0.0:
                    norm_penalty = self._normalization_penalty(
                        predictions,
                        X_batch.shape[0],
                        weights,
                    )
                    loss = loss + self.config.normalization_lambda * norm_penalty
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            batches += 1
        return total_loss / max(batches, 1)

    def _prepare_training_batch(
        self,
        X_batch: Tensor,
        y_batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        batch_size = X_batch.shape[0]
        m_aux = cfg.m_aux
        sampler_seed = self._next_sampler_seed()
        y_prime, sample_weights = self._draw_yprime((batch_size, m_aux), sampler_seed)
        adaptive = self._predict_adaptive_bandwidths(X_batch)
        kernels_per_bandwidth = []
        for idx, bandwidth in enumerate(self._bandwidths):
            if adaptive is None:
                effective_bw: float | Tensor = bandwidth
            else:
                head_scale = adaptive[:, idx].unsqueeze(1)
                effective_bw = head_scale * bandwidth
            kernels_per_bandwidth.append(
                kernel_h_torch(y_batch.unsqueeze(1), y_prime, effective_bw),
            )
        kernels = torch.stack(kernels_per_bandwidth, dim=-1)
        x_repeat = X_batch.repeat_interleave(m_aux, dim=0)
        yprime_flat = y_prime.reshape(-1, 1)
        features = torch.cat([x_repeat, yprime_flat], dim=1)
        targets = kernels.reshape(-1, len(self._bandwidths))
        weights = sample_weights.reshape(-1, 1)
        return features, targets, weights

    def _next_sampler_seed(self) -> int:
        seed = self.random_seed + self._sampler_call
        self._sampler_call += 1
        return seed

    def _draw_yprime(
        self,
        shape: tuple[int, int],
        seed: int,
    ) -> tuple[Tensor, Tensor]:
        method = self.config.sampler.lower()
        if method == "importance":
            if self._importance_sampler is None:
                msg = "Importance sampler not configured; ensure fit() was called first."
                raise RuntimeError(msg)
            return self._importance_sampler.draw(
                shape,
                seed=seed,
                device=self._device,
            )
        samples = sample_yprime(
            method,
            shape,
            seed=seed,
            device=self._device,
            value_range=(0.0, 1.0),
        )
        weights = torch.ones_like(samples)
        return samples, weights

    # ------------------------------------------------------------------ Inference
    def predict_density(
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating],
        *,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        self._ensure_fitted()
        return self._predict_density_internal(
            X,
            y_grid,
            head=head,
        )

    def _predict_density_internal(  # noqa: PLR0914
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating],
        *,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        assert self.x_scaler is not None
        assert self.y_scaler is not None
        assert self.model is not None
        X_arr: NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        y_grid_arr = self._validate_y_grid(y_grid)
        X_scaled = self.x_scaler.transform(X_arr)
        y_scaled = self.y_scaler.transform(y_grid_arr)
        x_tensor = torch.from_numpy(X_scaled).to(self._device)
        y_tensor = torch.from_numpy(y_scaled).to(self._device)
        grid_size = y_tensor.shape[0]

        self.model.eval()
        with torch.no_grad():
            x_expanded = x_tensor.unsqueeze(1).expand(-1, grid_size, -1)
            y_expanded = y_tensor.unsqueeze(0).expand(x_tensor.shape[0], -1).unsqueeze(-1)
            features = torch.cat([x_expanded, y_expanded], dim=-1).reshape(
                -1,
                x_tensor.shape[1] + 1,
            )
            preds = self.model(features).reshape(
                x_tensor.shape[0],
                grid_size,
                len(self._bandwidths),
            )
        pdf_scaled = preds.cpu().numpy().astype(np.float64, copy=False)
        scale = max(float(self.y_scaler.data_range_), 1e-8)
        density_heads = pdf_scaled / scale

        if not self.config.positive_output:
            density_heads = np.clip(density_heads, 0.0, None)

        normalization = np.trapezoid(density_heads, x=y_grid_arr, axis=1)
        normalization = np.clip(normalization, 1e-8, None)
        density_heads /= normalization[:, None, :]
        combined = self._combine_heads(density_heads, head=head)
        result: NDArray[np.float64] = combined.astype(np.float64, copy=False)
        return result

    def predict_cdf(
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating],
        *,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        pdf = self.predict_density(X, y_grid, head=head)
        y_grid_arr = self._validate_y_grid(y_grid)
        return self._cdf_from_pdf(pdf, y_grid_arr)

    def predict_quantile(
        self,
        X: NDArray[np.floating],
        q: NDArray[np.floating] | float,
        *,
        y_grid: NDArray[np.floating] | None = None,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        """Return empirical quantiles via inverse-CDF interpolation."""
        self._ensure_fitted()
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_arr.size == 0:
            msg = "Provide at least one quantile probability."
            raise ValueError(msg)
        if np.any((q_arr < 0.0) | (q_arr > 1.0)):
            msg = "Quantile probabilities must lie in [0, 1]."
            raise ValueError(msg)
        quantile_scalar = q_arr.size == 1 and np.ndim(q) == 0
        grid_arr = (
            self._validate_y_grid(y_grid) if y_grid is not None else self._default_y_grid()
        )
        cdf = self.predict_cdf(X, grid_arr, head=head)
        quantiles = self._quantiles_from_cdf(cdf, grid_arr, q_arr)
        if quantile_scalar:
            return quantiles[:, 0]
        return quantiles

    def predict_tail_prob(
        self,
        X: NDArray[np.floating],
        threshold: NDArray[np.floating] | float,
        *,
        side: Literal["right", "left"] = "right",
        y_grid: NDArray[np.floating] | None = None,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        """Return tail probabilities P(Y >= threshold) or P(Y <= threshold)."""
        self._ensure_fitted()
        side_norm = side.lower()
        if side_norm not in {"right", "left"}:
            msg = f"side must be 'right' or 'left', got {side!r}"
            raise ValueError(msg)
        grid_arr = (
            self._validate_y_grid(y_grid) if y_grid is not None else self._default_y_grid()
        )
        cdf = self.predict_cdf(X, grid_arr, head=head)
        thresholds = self._broadcast_to_samples(threshold, cdf.shape[0], "threshold")
        tail = np.empty(cdf.shape[0], dtype=np.float64)
        for idx, (cdf_row, thresh) in enumerate(zip(cdf, thresholds, strict=True)):
            value = float(np.interp(thresh, grid_arr, cdf_row, left=0.0, right=1.0))
            tail[idx] = 1.0 - value if side_norm == "right" else value
        return tail

    def expected_shortfall(
        self,
        X: NDArray[np.floating],
        alpha: float,
        *,
        side: Literal["right", "left"] = "right",
        y_grid: NDArray[np.floating] | None = None,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        """Compute tail conditional expectation (a.k.a. expected shortfall)."""
        self._ensure_fitted()
        alpha_arr = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if alpha_arr.size != 1:
            msg = "alpha must be a scalar probability."
            raise ValueError(msg)
        alpha_value = float(alpha_arr[0])
        if not 0.0 < alpha_value < 1.0:
            msg = "alpha must lie in (0, 1)."
            raise ValueError(msg)
        side_norm = side.lower()
        if side_norm not in {"right", "left"}:
            msg = f"side must be 'right' or 'left', got {side!r}"
            raise ValueError(msg)
        grid_arr = (
            self._validate_y_grid(y_grid) if y_grid is not None else self._default_y_grid()
        )
        pdf = self.predict_density(X, grid_arr, head=head)
        cdf = self._cdf_from_pdf(pdf, grid_arr)
        quantiles = self._quantiles_from_cdf(cdf, grid_arr, np.array([alpha_value], dtype=np.float64))
        thresholds = quantiles[:, 0]
        es = np.empty(pdf.shape[0], dtype=np.float64)
        for idx in range(pdf.shape[0]):
            numerator, mass = self._tail_moments(
                grid_arr,
                pdf[idx],
                thresholds[idx],
                greater=side_norm == "right",
            )
            if mass <= 1e-8:
                es[idx] = thresholds[idx]
            else:
                es[idx] = numerator / mass
        return es

    def sample(  # noqa: PLR0913
        self,
        X: NDArray[np.floating],
        n_samples: int,
        *,
        y_grid: NDArray[np.floating] | None = None,
        seed: int | None = None,
        refine_steps: int = 0,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        if n_samples <= 0:
            msg = "n_samples must be positive."
            raise ValueError(msg)
        if refine_steps < 0:
            msg = "refine_steps must be >= 0."
            raise ValueError(msg)
        grid_arr = (
            self._validate_y_grid(y_grid)
            if y_grid is not None
            else self._default_y_grid(mode="quantile")
        )
        pdf = self.predict_density(X, grid_arr, head=head)
        cdf = self._cdf_from_pdf(pdf, grid_arr)
        rng = np.random.default_rng(self.random_seed if seed is None else seed)
        samples = np.empty((X.shape[0], n_samples), dtype=np.float64)
        for idx in range(cdf.shape[0]):
            samples[idx] = self._inverse_transform_samples(
                grid_arr,
                pdf[idx],
                cdf[idx],
                n_samples,
                rng,
                refine_steps=refine_steps,
            )
        return samples

    # ------------------------------------------------------------------ Persistence
    def save(self, path: os.PathLike[str] | str) -> None:
        """Persist model weights, configuration, and scalers to disk."""
        self._ensure_fitted()
        assert self.model is not None
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), target / _MODEL_FILENAME)
        self._write_json(target / _CONFIG_FILENAME, self._config_to_dict(self.config))
        self._write_json(target / _SCALERS_FILENAME, self._scalers_to_dict())
        metadata: dict[str, Any] = {
            "random_seed": self.random_seed,
            "input_dim": self._infer_model_input_dim(),
            "bandwidths": self._bandwidths,
            "best_head_index": int(self._best_head_index),
        }
        y_reference = self._summarize_y_reference()
        if y_reference is not None:
            metadata["y_reference"] = y_reference
        self._write_json(target / _METADATA_FILENAME, metadata)

    @classmethod
    def load(
        cls,
        path: os.PathLike[str] | str,
        *,
        map_location: str | torch.device | None = None,
    ) -> CondensiteTorchCDE:
        """Load a saved estimator from disk."""
        base = Path(path)
        config_dict = cls._read_json(base / _CONFIG_FILENAME)
        scalers_dict = cls._read_json(base / _SCALERS_FILENAME)
        metadata = cls._read_json(base / _METADATA_FILENAME)
        config = cls._config_from_dict(config_dict)
        random_seed = int(metadata.get("random_seed", 0))
        estimator = cls(config=config, random_seed=random_seed)
        if map_location is not None:
            estimator._device = torch.device(map_location)
            estimator.config.device = estimator._device
        estimator._set_bandwidths(metadata.get("bandwidths"))
        estimator.x_scaler = cls._standard_scaler_from_dict(scalers_dict["x"])
        estimator.y_scaler = cls._minmax_scaler_from_dict(scalers_dict["y"])
        y_reference = metadata.get("y_reference")
        if y_reference is not None:
            estimator._y_train = np.asarray(y_reference, dtype=np.float64)
        input_dim = int(metadata["input_dim"])
        model_config = MLPRegressorConfig(
            input_dim=input_dim,
            hidden_sizes=estimator.config.hidden_sizes,
            activation=estimator.config.activation,
            dropout=estimator.config.dropout,
            positive_output=estimator.config.positive_output,
        )
        estimator.model = MLPRegressor(model_config).to(estimator._device)
        state_dict = torch.load(base / _MODEL_FILENAME, map_location=estimator._device)
        estimator.model.load_state_dict(state_dict)
        estimator._best_head_index = int(metadata.get("best_head_index", 0))
        estimator.training_history.clear()
        estimator._fitted = True
        return estimator

    def _default_y_grid(self, num_points: int = 128, mode: str = "quantile") -> NDArray[np.float64]:
        if self._y_train is not None:
            try:
                return make_y_grid(self._y_train, grid_size=num_points, mode=mode)
            except ValueError:
                pass
        if self.y_scaler is None:
            msg = "Target scaler is not available; fit the estimator first."
            raise RuntimeError(msg)
        return np.linspace(
            self.y_scaler.min_,
            self.y_scaler.max_,
            num_points,
            dtype=np.float64,
        )

    @staticmethod
    def _cdf_from_pdf(
        pdf: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        diffs = np.diff(y_grid)
        trapezoids = 0.5 * (pdf[:, 1:] + pdf[:, :-1]) * diffs
        cdf = np.concatenate([np.zeros((pdf.shape[0], 1)), np.cumsum(trapezoids, axis=1)], axis=1)
        cdf = np.maximum.accumulate(cdf, axis=1)
        norm = np.clip(cdf[:, -1:], 1e-8, None)
        cdf = np.clip(cdf / norm, 0.0, 1.0)
        cdf[:, -1] = 1.0
        return cdf.astype(np.float64, copy=False)

    def _inverse_transform_samples(  # noqa: PLR0913
        self,
        y_grid: NDArray[np.float64],
        pdf_row: NDArray[np.float64],
        cdf_row: NDArray[np.float64],
        n_samples: int,
        rng: np.random.Generator,
        *,
        refine_steps: int,
    ) -> NDArray[np.float64]:
        draws = rng.random(n_samples)
        indices = np.searchsorted(cdf_row, draws, side="right")
        indices = np.clip(indices, 1, y_grid.size - 1)
        y_low = y_grid[indices - 1]
        y_high = y_grid[indices]
        cdf_low = cdf_row[indices - 1]
        cdf_high = cdf_row[indices]
        pdf_low = pdf_row[indices - 1]
        pdf_high = pdf_row[indices]

        widths = np.maximum(y_high - y_low, 1e-12)
        cdf_span = np.maximum(cdf_high - cdf_low, 1e-12)
        rel = np.clip((draws - cdf_low) / cdf_span, 0.0, 1.0)
        target = np.clip(draws - cdf_low, 0.0, cdf_span)

        if refine_steps > 0:
            rel = self._refine_relative_positions(
                rel,
                target,
                widths,
                pdf_low,
                pdf_high,
                refine_steps,
            )

        return y_low + rel * widths

    @staticmethod
    def _refine_relative_positions(  # noqa: PLR0913, PLR0917
        rel: NDArray[np.float64],
        target_mass: NDArray[np.float64],
        widths: NDArray[np.float64],
        pdf_low: NDArray[np.float64],
        pdf_high: NDArray[np.float64],
        refine_steps: int,
    ) -> NDArray[np.float64]:
        t = rel.copy()
        a = pdf_low
        b = pdf_high - pdf_low
        width_safe = np.maximum(widths, 1e-12)
        s = target_mass / width_safe
        for _ in range(refine_steps):
            denom = np.maximum(a + b * t, 1e-12)
            value = a * t + 0.5 * b * t * t - s
            t = np.clip(t - value / denom, 0.0, 1.0)
        return t

    def _build_bandwidth_net(self, input_dim: int) -> MLPRegressor:
        config = MLPRegressorConfig(
            input_dim=input_dim,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            dropout=self.config.dropout,
            positive_output=False,
            output_dim=len(self._bandwidths),
        )
        return MLPRegressor(config).to(self._device)

    def _combine_heads(
        self,
        density_heads: NDArray[np.float64],
        *,
        head: int | str | None,
    ) -> NDArray[np.float64]:
        num_heads = density_heads.shape[-1]
        if num_heads == 1:
            return density_heads[..., 0]
        resolved = head if head is not None else self.config.bandwidth_strategy
        if isinstance(resolved, str):
            option = resolved.lower()
            if option == "mean":
                return density_heads.mean(axis=-1)
            if option == "best":
                index = int(self._best_head_index)
            elif option == "first":
                index = 0
            else:
                msg = f"Unknown head selection '{resolved}'."
                raise ValueError(msg)
        else:
            index = int(resolved)
        if index < 0 or index >= num_heads:
            msg = f"Head index {index} is out of range for {num_heads} heads."
            raise ValueError(msg)
        return density_heads[..., index]

    def _normalization_penalty(
        self,
        predictions: Tensor,
        batch_size: int,
        sample_weights: Tensor,
    ) -> Tensor:
        m_aux = self.config.m_aux
        if m_aux <= 0:
            return predictions.new_tensor(0.0)
        num_heads = predictions.shape[-1]
        if batch_size * m_aux != predictions.shape[0]:
            msg = (
                "Prediction tensor shape does not match batch size and m_aux. "
                f"Expected {batch_size * m_aux}, received {predictions.shape[0]} rows."
            )
            raise RuntimeError(msg)
        reshaped = predictions.reshape(batch_size, m_aux, num_heads)
        weights = sample_weights.reshape(batch_size, m_aux, 1)
        weights = weights / torch.clamp(weights.mean(dim=1, keepdim=True), min=1e-6)
        integrals = torch.sum(reshaped * weights, dim=1) / torch.clamp(
            weights.sum(dim=1),
            min=1e-6,
        )
        penalty = (integrals - 1.0) ** 2
        return penalty.mean()

    def _weighted_mse_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        sample_weights: Tensor,
    ) -> Tensor:
        squared_error = (predictions - targets) ** 2
        weighted = squared_error * sample_weights
        return weighted.mean()

    def _predict_adaptive_bandwidths(self, X_batch: Tensor) -> Tensor | None:
        if self.bandwidth_net is None:
            return None
        raw = self.bandwidth_net(X_batch)
        return F.softplus(raw) + 1e-6

    def _determine_best_head(
        self,
        val_eval_data: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
        y_grid: NDArray[np.float64] | None,
    ) -> None:
        if val_eval_data is None or y_grid is None or len(self._bandwidths) <= 1:
            self._best_head_index = 0
            return
        X_val_eval, y_val_eval = val_eval_data
        best_idx = 0
        best_value = math.inf
        metric_name = self.config.monitor_metric
        for idx in range(len(self._bandwidths)):
            pdf = self._predict_density_internal(X_val_eval, y_grid, head=idx)
            if metric_name == "val_nll":
                metric_value = nll_from_pdf(y_val_eval, y_grid, pdf)
            else:
                cdf = self._cdf_from_pdf(pdf, y_grid)
                metric_value = crps_from_cdf(y_val_eval, y_grid, cdf)
            if metric_value < best_value:
                best_value = metric_value
                best_idx = idx
        self._best_head_index = best_idx

    def _configure_importance_sampler(self, y_scaled: NDArray[np.float32]) -> None:
        if self.config.sampler.lower() != "importance":
            self._importance_sampler = None
            return
        self._importance_sampler = ImportanceSampler.from_array(y_scaled)

    @staticmethod
    def _quantiles_from_cdf(
        cdf: NDArray[np.float64],
        y_grid: NDArray[np.float64],
        probs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        quantiles = np.empty((cdf.shape[0], probs.size), dtype=np.float64)
        for col, prob in enumerate(probs):
            quantiles[:, col] = np.array(
                [
                    np.interp(prob, row, y_grid, left=y_grid[0], right=y_grid[-1])
                    for row in cdf
                ],
                dtype=np.float64,
            )
        return quantiles

    @staticmethod
    def _broadcast_to_samples(
        values: NDArray[np.floating] | float,
        n_samples: int,
        name: str,
    ) -> NDArray[np.float64]:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            msg = f"{name} must contain at least one value."
            raise ValueError(msg)
        if arr.size == 1:
            return np.full(n_samples, float(arr[0]), dtype=np.float64)
        if arr.size != n_samples:
            msg = f"{name} must be scalar or match the number of samples ({n_samples})."
            raise ValueError(msg)
        return arr.astype(np.float64, copy=False)

    @staticmethod
    def _tail_moments(
        y_grid: NDArray[np.float64],
        pdf_row: NDArray[np.float64],
        threshold: float,
        *,
        greater: bool,
    ) -> tuple[float, float]:
        grid = np.asarray(y_grid, dtype=np.float64)
        pdf = np.asarray(pdf_row, dtype=np.float64)
        if greater:
            if threshold >= grid[-1]:
                return 0.0, 0.0
            if threshold <= grid[0]:
                y_vals = grid
                pdf_vals = pdf
            else:
                idx = int(np.searchsorted(grid, threshold, side="left"))
                interp = float(np.interp(threshold, grid, pdf))
                y_vals = np.concatenate(([threshold], grid[idx:]))
                pdf_vals = np.concatenate(([interp], pdf[idx:]))
        else:
            if threshold <= grid[0]:
                return 0.0, 0.0
            if threshold >= grid[-1]:
                y_vals = grid
                pdf_vals = pdf
            else:
                idx = int(np.searchsorted(grid, threshold, side="right"))
                interp = float(np.interp(threshold, grid, pdf))
                y_vals = np.concatenate((grid[:idx], [threshold]))
                pdf_vals = np.concatenate((pdf[:idx], [interp]))
        mass = float(np.trapezoid(pdf_vals, x=y_vals))
        moment = float(np.trapezoid(pdf_vals * y_vals, x=y_vals))
        return moment, mass

    def _summarize_y_reference(self) -> list[float] | None:
        if self._y_train is None or self._y_train.size == 0:
            return None
        arr = np.asarray(self._y_train, dtype=np.float64).reshape(-1)
        if arr.size <= _Y_REFERENCE_POINTS:
            return arr.tolist()
        quantiles = np.linspace(0.0, 1.0, _Y_REFERENCE_POINTS)
        summary = np.quantile(arr, quantiles, method="linear")
        return summary.tolist()

    def _scalers_to_dict(self) -> dict[str, Any]:
        assert self.x_scaler is not None
        assert self.y_scaler is not None
        return {
            "x": {
                "mean": self.x_scaler.mean_.tolist(),
                "scale": self.x_scaler.scale_.tolist(),
                "eps": self.x_scaler.eps,
            },
            "y": {
                "min": self.y_scaler.min_,
                "max": self.y_scaler.max_,
                "eps": self.y_scaler.eps,
            },
        }

    @staticmethod
    def _standard_scaler_from_dict(data: dict[str, Any]) -> StandardScaler:
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(data["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(data["scale"], dtype=np.float64)
        scaler.eps = float(data.get("eps", scaler.eps))
        scaler.fitted_ = True
        return scaler

    @staticmethod
    def _minmax_scaler_from_dict(data: dict[str, Any]) -> MinMaxScaler1D:
        scaler = MinMaxScaler1D()
        scaler.min_ = float(data["min"])
        scaler.max_ = float(data["max"])
        scaler.eps = float(data.get("eps", scaler.eps))
        scaler.fitted_ = True
        return scaler

    def _config_to_dict(self, config: CondensiteTorchCDEConfig) -> dict[str, Any]:
        data = asdict(config)
        activation = data.get("activation")
        data["activation"] = self._serialize_activation(activation)
        device_value = data.get("device")
        if isinstance(device_value, torch.device):
            data["device"] = str(device_value)
        return data

    @classmethod
    def _config_from_dict(cls, data: dict[str, Any]) -> CondensiteTorchCDEConfig:
        loaded = dict(data)
        activation_name = loaded.get("activation")
        loaded["activation"] = None
        config = CondensiteTorchCDEConfig(**loaded)
        config.activation = cls._deserialize_activation(activation_name)
        return config

    @staticmethod
    def _serialize_activation(
        activation: ActivationFactory | nn.Module | None,
    ) -> str | None:
        if activation is None:
            return None
        if isinstance(activation, nn.Module):
            return activation.__class__.__name__
        name = getattr(activation, "__name__", None)
        return str(name) if name is not None else repr(activation)

    @staticmethod
    def _deserialize_activation(name: str | None) -> ActivationFactory | nn.Module | None:
        if name is None:
            return None
        normalized = name.replace("-", "").replace("_", "").lower()
        factory_cls = _ACTIVATION_REGISTRY.get(normalized)
        if factory_cls is None:
            msg = f"Unsupported activation identifier '{name}'."
            raise ValueError(msg)
        return factory_cls

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _infer_model_input_dim(self) -> int:
        assert self.model is not None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                return int(module.in_features)
        msg = "Unable to infer model input dimension from layers."
        raise RuntimeError(msg)

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

    def _set_bandwidths(self, override: Sequence[float] | None = None) -> None:
        self._bandwidths = self._resolve_bandwidths(override)

    def _resolve_bandwidths(self, override: Sequence[float] | None = None) -> list[float]:
        if override is not None:
            values = list(override)
        elif self.config.bandwidths:
            values = list(self.config.bandwidths)
        else:
            values = [self.config.bandwidth]
        processed = [float(val) for val in values if val is not None]
        if not processed:
            msg = "At least one bandwidth must be provided."
            raise ValueError(msg)
        if any(val <= 0.0 for val in processed):
            msg = "Bandwidth values must be positive."
            raise ValueError(msg)
        return processed


__all__: tuple[str, ...] = ("CondensiteTorchCDE", "CondensiteTorchCDEConfig")
