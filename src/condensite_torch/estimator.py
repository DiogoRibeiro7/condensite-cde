"""Condensité estimator with streamed auxiliary sampling and torch training."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess  # noqa: S404
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from condensite_cde.grids import GridMode, make_y_grid

from . import local_grids
from .aux_sampling import ImportanceSampler, sample_yprime
from .kernels import KernelSpec, get_kernel_spec
from .losses import LossSpec, get_loss_spec
from .metrics import crps_from_cdf, nll_from_pdf
from .models import ActivationFactory, MLPRegressor, MLPRegressorConfig
from .preprocessing import TabularPreprocessor, TabularPreprocessorConfig
from .scalers import MinMaxScaler1D, StandardScaler
from .validation import SchemaConstraints, validate_inputs


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
    aux_chunk_size: int | None = None
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    sampler: str = "sobol"
    amp: bool = False
    kernel: Literal["gaussian", "epanechnikov"] = "gaussian"
    loss: Literal["mse", "mae"] = "mse"
    val_fraction: float = 0.1
    device: str | torch.device = "cpu"
    monitor_metric: Literal["val_crps", "val_nll"] = "val_crps"
    epistemic_mode: Literal["none", "mc_dropout", "ensemble"] = "none"
    mc_samples: int = 10
    ensemble_size: int = 1
    reproducible: bool = False
    inference_batch_size: int | None = None
    inference_grid_chunk_size: int | None = None
    input_schema: SchemaConstraints | None = None
    preprocessor: TabularPreprocessorConfig | None = field(
        default_factory=TabularPreprocessorConfig,
    )


_MODEL_FILENAME = "model.pt"
_CONFIG_FILENAME = "config.json"
_SCALERS_FILENAME = "scalers.json"
_METADATA_FILENAME = "metadata.json"
_BANDWIDTH_MODEL_FILENAME = "bandwidth_net.pt"
_PREPROCESSOR_FILENAME = "preprocessor.json"
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
_LOCAL_GRID_CACHE_SIZE = 4
_GRID_DIMENSION = 2
_MIN_GRID_POINTS = 2
_SMALL_MASS = 1e-8


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
        self._kernel_spec: KernelSpec = get_kernel_spec(self.config.kernel)
        self._loss_spec: LossSpec = get_loss_spec(self.config.loss)
        self.bandwidth_net: MLPRegressor | None = None
        self.x_scaler: StandardScaler | None = None
        self.y_scaler: MinMaxScaler1D | None = None
        self.model: MLPRegressor | None = None
        self.training_history: list[dict[str, float]] = []
        self._fitted = False
        self._y_train: NDArray[np.float64] | None = None
        self._best_head_index = 0
        self._importance_sampler: ImportanceSampler | None = None
        self._best_epoch: int = -1
        self._restored_best_epoch: int | None = None
        self._current_epoch: int = -1
        self._model_card: dict[str, Any] | None = None
        self._data_schema: dict[str, Any] | None = None
        self._version_info: dict[str, Any] | None = None
        self._schema_constraints: SchemaConstraints | None = (
            copy.deepcopy(self.config.input_schema)
            if self.config.input_schema is not None
            else None
        )
        self.preprocessor: TabularPreprocessor | None = None
        self._local_grid_cache: OrderedDict[
            tuple[str, int, float, float, float, str | None, str | None],
            NDArray[np.float64],
        ] = OrderedDict()

    # ------------------------------------------------------------------ Training
    def fit(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        *,
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> CondensiteTorchCDE:
        """Train the estimator with streamed auxiliary samples.

        Args:
            X (NDArray[np.floating]): Training features shaped `(n_samples, n_features)`.
            y (NDArray[np.floating]): Training targets shaped `(n_samples,)`.
            X_val (NDArray[np.floating] | None): Optional validation features to
                skip internal split.
            y_val (NDArray[np.floating] | None): Optional validation targets matching
                `X_val`.

        Returns:
            CondensiteTorchCDE: The fitted estimator (`self`).

        Raises:
            ValueError: If input shapes mismatch or validation arrays have inconsistent lengths.
            RuntimeError: If preprocessing fails due to incompatible inputs.

        Side Effects:
            Mutates internal state (model weights, scalers, metadata) and may write AMP warnings.

        Complexity:
            O(epochs * n_samples * m_aux) forward/backward passes, dominated by kernel targets.
        """
        X_arr = np.asarray(X, dtype=object)
        y_arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64).reshape(-1)
        schema = self._schema_for_validation(include_targets=True)
        validate_inputs(X_arr, y_arr, schema=schema, context="fit")

        self._local_grid_cache.clear()
        self._set_bandwidths()
        self._y_train = y_arr.copy()
        self._data_schema = self._summarize_data_schema(X_arr, y_arr)
        self._version_info = self._environment_versions()
        start_time = time.perf_counter()

        if self.config.reproducible:
            self._seed_everything()
        else:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        if self.config.preprocessor is not None:
            self.preprocessor = TabularPreprocessor(self.config.preprocessor)
        else:
            self.preprocessor = None
        X_features = self._fit_preprocessor(X_arr)
        self.y_scaler = MinMaxScaler1D().fit(y_arr)
        derived_schema = self._schema_from_training(X_arr, y_arr)
        self._schema_constraints = self._merge_schema_constraints(
            self._schema_constraints,
            derived_schema,
        )

        X_scaled: NDArray[np.float32] = X_features.astype(np.float32)
        y_scaled: NDArray[np.float32] = self.y_scaler.transform(y_arr)

        val_eval_data: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        if X_val is None or y_val is None:
            X_train_scaled, y_train_scaled, val_idx = self._train_val_split(X_scaled, y_scaled)
            if val_idx is not None and val_idx.size > 0:
                val_eval_data = (X_arr[val_idx], y_arr[val_idx])
        else:
            X_val_arr = np.asarray(X_val, dtype=object)
            y_val_arr = np.asarray(y_val, dtype=np.float64).reshape(-1)
            if X_val_arr.shape[0] != y_val_arr.shape[0]:
                msg = "Validation X and y must match in the first dimension."
                raise ValueError(msg)
            self._transform_with_preprocessor(X_val_arr)  # Validate shape
            self.y_scaler.transform(y_val_arr)
            X_train_scaled, y_train_scaled = X_scaled, y_scaled
            if X_val_arr.size > 0:
                val_eval_data = (X_val_arr, y_val_arr)
            else:
                val_eval_data = None

        train_loader = self._build_loader(X_train_scaled, y_train_scaled, shuffle=True)
        self._configure_importance_sampler(y_train_scaled)

        x_dim = X_scaled.shape[1]
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
        best_model_state: dict[str, Tensor] | None = None
        best_bandwidth_state: dict[str, Tensor] | None = None
        best_metric = math.inf
        epochs_without_improvement = 0
        self.training_history.clear()
        self._best_epoch = -1
        self._restored_best_epoch = None
        monitor_metric = self.config.monitor_metric
        valid_monitors = {"val_crps", "val_nll", "val_integral_error"}
        if monitor_metric not in valid_monitors:
            msg = f"monitor_metric must be one of {sorted(valid_monitors)}, got {monitor_metric!r}"
            raise ValueError(msg)
        val_grid: NDArray[np.float64] | None = None

        for epoch in range(self.config.epochs):
            self._current_epoch = epoch
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
                val_mass = np.trapezoid(pdf_val, x=val_grid, axis=1)
                val_integral_error = float(np.mean(np.abs(val_mass - 1.0)))
                record["val_crps"] = val_crps
                record["val_nll"] = val_nll
                record["val_integral_error"] = val_integral_error
                monitor_value = record[monitor_metric]
            self.training_history.append(record)

            if monitor_value < best_metric - 1e-6:
                best_metric = monitor_value
                best_model_state = copy.deepcopy(self.model.state_dict())
                if self.bandwidth_net is not None:
                    best_bandwidth_state = copy.deepcopy(self.bandwidth_net.state_dict())
                self._best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.patience:
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if self.bandwidth_net is not None and best_bandwidth_state is not None:
                self.bandwidth_net.load_state_dict(best_bandwidth_state)
            self._restored_best_epoch = self._best_epoch
        if val_eval_data is not None:
            eval_grid = val_grid if val_grid is not None else self._default_y_grid()
        else:
            eval_grid = None
        self._determine_best_head(val_eval_data, eval_grid)
        self._fitted = True
        training_time = time.perf_counter() - start_time
        best_metric_value = None if not math.isfinite(best_metric) else float(best_metric)
        self._model_card = self._build_model_card(
            schema=self._data_schema,
            training_time_seconds=training_time,
            best_metric=best_metric_value,
            monitor_metric=monitor_metric,
        )
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
            amp_context = torch.autocast("cuda") if amp_enabled else nullcontext()
            with amp_context:
                loss = self._compute_batch_loss(
                    X_batch,
                    y_batch,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            batches += 1
        return total_loss / max(batches, 1)

    def _compute_batch_loss(  # noqa: PLR0914
        self,
        X_batch: Tensor,
        y_batch: Tensor,
    ) -> Tensor:
        assert self.model is not None
        batch_inputs = X_batch.to(self._device)
        targets_batch = y_batch.to(self._device)
        batch_size = batch_inputs.shape[0]
        sampler_seed = self._next_sampler_seed()
        y_prime, sample_weights = self._draw_yprime((batch_size, self.config.m_aux), sampler_seed)
        adaptive = self._predict_adaptive_bandwidths(batch_inputs)
        chunk_size = self._effective_aux_chunk_size()
        total_weighted_error = batch_inputs.new_tensor(0.0)
        total_entries = 0
        norm_acc = (
            _NormalizationAccumulator(batch_size, len(self._bandwidths), self._device)
            if self.config.normalization_lambda > 0.0
            else None
        )
        for y_chunk, weights_chunk in self._iter_aux_chunks(y_prime, sample_weights, chunk_size):
            features = self._combine_features(batch_inputs, y_chunk)
            target_chunk = self._compute_kernel_targets(targets_batch, y_chunk, adaptive)
            predictions = self.model(features)
            weights_flat = weights_chunk.reshape(-1, 1)
            per_entry = self._loss_spec.elementwise_fn(predictions, target_chunk)
            weighted = per_entry * weights_flat
            total_weighted_error += weighted.sum()
            total_entries += weights_flat.numel()
            if norm_acc is not None:
                preds_reshaped = predictions.view(batch_size, -1, len(self._bandwidths))
                norm_acc.accumulate(preds_reshaped, weights_chunk.unsqueeze(-1))
        loss = total_weighted_error / max(total_entries, 1)
        if norm_acc is None:
            return loss
        penalty = norm_acc.penalty()
        return loss + self.config.normalization_lambda * penalty

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
        y_grid: NDArray[np.floating] | None,
        *,
        head: int | str | None = None,
        use_local_grid: bool = False,
        local_grid_params: dict[str, float | int] | None = None,
    ) -> NDArray[np.float64]:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=object)
        schema = self._schema_for_validation(include_targets=False)
        validate_inputs(X_arr, None, schema=schema, context="predict_density")
        grid = y_grid
        if grid is None and use_local_grid:
            grid = self._build_local_grids(X_arr, local_grid_params)
        if grid is None:
            grid = self._default_y_grid()
        return self._predict_density_with_epistemic(
            X_arr,
            grid,
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
        y_scaler = self.y_scaler
        model = self.model
        X_arr = np.asarray(X, dtype=object)
        grid_arr, local = self._process_y_grid(X_arr, y_grid)
        X_scaled = self._transform_with_preprocessor(X_arr).astype(np.float32, copy=False)
        x_tensor = torch.from_numpy(X_scaled).to(self._device)
        if local:
            return self._predict_density_local(x_tensor, grid_arr, head=head)
        y_scaled = self.y_scaler.transform(grid_arr).astype(np.float32, copy=False)
        y_tensor_scaled = torch.from_numpy(y_scaled).to(self._device)
        y_tensor = torch.from_numpy(grid_arr.astype(np.float32, copy=False)).to(self._device)

        row_batch = self._inference_row_batch_size(x_tensor.shape[0])
        outputs: list[Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, x_tensor.shape[0], row_batch):
                end = min(start + row_batch, x_tensor.shape[0])
                batch = x_tensor[start:end]
                outputs.append(
                    self._predict_density_row_batch(
                        batch,
                        y_tensor_scaled,
                        y_tensor,
                    ),
                )
        normalized = (
            torch.cat(outputs, dim=0)
            if outputs
            else torch.zeros(
                (0, y_tensor.shape[0], len(self._bandwidths)),
                device=self._device,
            )
        )
        density_heads = normalized.cpu().numpy().astype(np.float64, copy=False)
        combined = self._combine_heads(density_heads, head=head)
        result: NDArray[np.float64] = combined.astype(np.float64, copy=False)
        return result

    def _predict_density_with_epistemic(
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating],
        *,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        mode = self.config.epistemic_mode
        if mode == "mc_dropout":
            return self._predict_density_mc_dropout(X, y_grid, head=head)
        return self._predict_density_internal(X, y_grid, head=head)

    def _predict_density_mc_dropout(
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating],
        *,
        head: int | str | None = None,
    ) -> NDArray[np.float64]:
        """Estimate density by averaging multiple dropout-enabled forward passes."""
        self._ensure_fitted()
        assert self.model is not None
        samples = max(1, int(self.config.mc_samples))
        X_arr = np.asarray(X, dtype=object)
        schema = self._schema_for_validation(include_targets=False)
        validate_inputs(X_arr, None, schema=schema, context="predict_density")
        y_grid_arr = self._validate_y_grid(y_grid)
        original_mode = self.model.training
        self.model.train(True)
        densities: list[NDArray[np.float64]] = []
        try:
            for _ in range(samples):
                pdf = self._predict_density_internal(X_arr, y_grid_arr, head=head)
                densities.append(pdf)
        finally:
            self.model.train(original_mode)
        stacked = np.stack(densities, axis=0)
        return cast(NDArray[np.float64], stacked.mean(axis=0)).astype(np.float64, copy=False)

    def _predict_density_local(  # noqa: PLR0914
        self,
        x_tensor: Tensor,
        y_grids: NDArray[np.float64],
        *,
        head: int | str | None,
    ) -> NDArray[np.float64]:
        assert self.y_scaler is not None
        assert self.model is not None
        y_scaler = self.y_scaler
        model = self.model
        n_rows, grid_size = y_grids.shape
        y_scaled = y_scaler.transform(y_grids.reshape(-1)).reshape(n_rows, grid_size)
        y_scaled_tensor = torch.from_numpy(y_scaled.astype(np.float32)).to(self._device)
        features: list[Tensor] = []
        counts: list[int] = []
        for row in range(n_rows):
            count = grid_size
            counts.append(count)
            x_row = x_tensor[row].unsqueeze(0).expand(count, -1)
            y_row_scaled = y_scaled_tensor[row].unsqueeze(1)
            features.append(torch.cat([x_row, y_row_scaled], dim=1))
        concatenated = torch.cat(features, dim=0)
        scale = max(float(y_scaler.data_range_), 1e-8)
        model.eval()
        with torch.no_grad():
            preds = torch.clamp(model(concatenated) / scale, min=1e-12)
        start = 0
        row_heads: list[NDArray[np.float64]] = []
        for row, count in enumerate(counts):
            chunk = preds[start : start + count]
            y_row = torch.from_numpy(y_grids[row].astype(np.float32)).to(self._device)
            normalized = self._normalize_pdf_heads_row(chunk, y_row)
            row_heads.append(normalized.cpu().numpy().astype(np.float64, copy=False))
            start += count
        density_heads = np.stack(row_heads, axis=0)
        combined = self._combine_heads(density_heads, head=head)
        return combined.astype(np.float64, copy=False)

    def predict_cdf(
        self,
        X: NDArray[np.floating],
        y_grid: NDArray[np.floating] | None,
        *,
        head: int | str | None = None,
        use_local_grid: bool = False,
        local_grid_params: dict[str, float | int] | None = None,
    ) -> NDArray[np.float64]:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=object)
        schema = self._schema_for_validation(include_targets=False)
        validate_inputs(X_arr, None, schema=schema, context="predict_cdf")
        grid = y_grid
        if grid is None and use_local_grid:
            grid = self._build_local_grids(X_arr, local_grid_params)
        if grid is None:
            grid = self._default_y_grid()
        pdf = self.predict_density(
            X_arr,
            grid,
            head=head,
        )
        if isinstance(grid, np.ndarray) and grid.ndim == _GRID_DIMENSION:
            cdfs = np.empty_like(pdf)
            for idx in range(pdf.shape[0]):
                cdfs[idx] = self._cdf_from_pdf(pdf[idx : idx + 1], grid[idx])[0]
            return cdfs
        y_grid_arr = self._validate_y_grid(grid)
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
        grid_arr = self._validate_y_grid(y_grid) if y_grid is not None else self._default_y_grid()
        cdf = self.predict_cdf(X, grid_arr, head=head)
        quantiles = self._quantiles_from_cdf(cdf, grid_arr, q_arr)
        if quantile_scalar:
            return quantiles[:, 0]
        return quantiles

    def predict_interval(
        self,
        X: NDArray[np.floating],
        coverage: float = 0.9,
        *,
        y_grid: NDArray[np.floating] | None = None,
        head: int | str | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return central predictive interval (lo, hi) for the desired coverage."""
        self._ensure_fitted()
        if not 0.0 < coverage < 1.0:
            msg = "coverage must be in the open interval (0, 1)."
            raise ValueError(msg)
        tail_mass = (1.0 - float(coverage)) / 2.0
        probs = np.array([max(tail_mass, 0.0), min(1.0 - tail_mass, 1.0)], dtype=np.float64)
        quantiles = self.predict_quantile(X, probs, y_grid=y_grid, head=head)
        return quantiles[:, 0], quantiles[:, 1]

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
        grid_arr = self._validate_y_grid(y_grid) if y_grid is not None else self._default_y_grid()
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
        grid_arr = self._validate_y_grid(y_grid) if y_grid is not None else self._default_y_grid()
        pdf = self.predict_density(X, grid_arr, head=head)
        cdf = self._cdf_from_pdf(pdf, grid_arr)
        quantiles = self._quantiles_from_cdf(
            cdf,
            grid_arr,
            np.array([alpha_value], dtype=np.float64),
        )
        thresholds = quantiles[:, 0]
        es = np.empty(pdf.shape[0], dtype=np.float64)
        for idx in range(pdf.shape[0]):
            numerator, mass = self._tail_moments(
                grid_arr,
                pdf[idx],
                thresholds[idx],
                greater=side_norm == "right",
            )
            if mass <= _SMALL_MASS:
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

    def evaluate(  # noqa: PLR0913
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        *,
        y_grid: NDArray[np.floating] | None = None,
        head: int | str | None = None,
        use_local_grid: bool = False,
        local_grid_params: dict[str, float | int] | None = None,
    ) -> dict[str, float]:
        """Compute NLL/CRPS/integral error metrics for (X, y).

        Args:
            X (NDArray[np.floating]): Features to evaluate.
            y (NDArray[np.floating]): Observed targets aligned with `X`.
            y_grid (NDArray[np.floating] | None): Optional custom grid; defaults to training grid.
            head (int | str | None): Which bandwidth head to use.
            use_local_grid (bool): Whether to build per-row grids for evaluation.
            local_grid_params (dict[str, float | int] | None): Extra kwargs forwarded
                to local grid builder.

        Returns:
            dict[str, float]: Dictionary with `nll`, `crps`, and `integral_error`.

        Raises:
            ValueError: If shapes mismatch.

        Side Effects:
            None.

        Complexity:
            O(n_samples * grid_size).
        """
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=object)
        y_arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64).reshape(-1)
        schema = self._schema_for_validation(include_targets=True)
        validate_inputs(X_arr, y_arr, schema=schema, context="evaluate")
        grid = y_grid
        if grid is None and use_local_grid:
            grid = self._build_local_grids(X_arr, local_grid_params)
        if grid is None:
            grid = self._default_y_grid()
        pdf = self.predict_density(
            X_arr,
            grid,
            head=head,
            use_local_grid=use_local_grid,
            local_grid_params=local_grid_params,
        )
        if isinstance(grid, np.ndarray) and grid.ndim == _GRID_DIMENSION:
            cdf = np.empty_like(pdf)
            for idx in range(pdf.shape[0]):
                cdf[idx] = self._cdf_from_pdf(pdf[idx : idx + 1], grid[idx])[0]
            grid_arr = grid
        else:
            grid_arr = self._validate_y_grid(grid)
            cdf = self._cdf_from_pdf(pdf, grid_arr)
        nll_value = nll_from_pdf(y_arr, grid_arr, pdf)
        crps_value = crps_from_cdf(y_arr, grid_arr, cdf)
        masses = np.trapezoid(pdf, x=grid_arr, axis=1)
        integral_error = float(np.mean(np.abs(masses - 1.0)))
        return {
            "nll": float(nll_value),
            "crps": float(crps_value),
            "integral_error": integral_error,
        }

    def model_card(self) -> dict[str, Any]:
        """Return the recorded reproducibility metadata.

        Returns:
            dict[str, Any]: Deep copy of the model card with config hash,
                schema, and training summary.

        Raises:
            RuntimeError: If `fit` has not been called and no metadata exists yet.

        Side Effects:
            None.

        Complexity:
            O(size(card)) due to the defensive deep copy.
        """
        if self._model_card is None:
            msg = "Call fit() before requesting the model card."
            raise RuntimeError(msg)
        return copy.deepcopy(self._model_card)

    # ------------------------------------------------------------------ Persistence
    def save(self, path: os.PathLike[str] | str) -> None:
        """Persist model weights, configuration, scalers, and metadata to disk.

        Args:
            path (os.PathLike | str): Target directory for artifacts.

        Returns:
            None.

        Raises:
            RuntimeError: If `fit` has not been called.

        Side Effects:
            Writes multiple files (model weights, JSON metadata) under `path`.

        Complexity:
            O(params) for serializing weights and scalers.
        """
        self._ensure_fitted()
        assert self.model is not None
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), target / _MODEL_FILENAME)
        self._write_json(target / _CONFIG_FILENAME, self._config_to_dict(self.config))
        self._write_json(target / _SCALERS_FILENAME, self._scalers_to_dict())
        preprocessor_payload = self._preprocessor_to_dict()
        if preprocessor_payload is not None:
            self._write_json(target / _PREPROCESSOR_FILENAME, preprocessor_payload)
        metadata: dict[str, Any] = {
            "random_seed": self.random_seed,
            "input_dim": self._infer_model_input_dim(),
            "bandwidths": self._bandwidths,
            "best_head_index": int(self._best_head_index),
            "best_epoch": int(self._best_epoch),
            "restored_best_epoch": (
                None if self._restored_best_epoch is None else int(self._restored_best_epoch)
            ),
        }
        if self._model_card is not None:
            metadata["model_card"] = self._model_card
        if self._schema_constraints is not None:
            metadata["schema_constraints"] = self._schema_constraints_to_dict(
                self._schema_constraints,
            )
        if self.bandwidth_net is not None:
            torch.save(self.bandwidth_net.state_dict(), target / _BANDWIDTH_MODEL_FILENAME)
            metadata["has_bandwidth_net"] = True
        y_reference = self._summarize_y_reference()
        if y_reference is not None:
            metadata["y_reference"] = y_reference
        self._write_json(target / _METADATA_FILENAME, metadata)

    @classmethod
    def load(  # noqa: PLR0914
        cls,
        path: os.PathLike[str] | str,
        *,
        map_location: str | torch.device | None = None,
    ) -> CondensiteTorchCDE:
        """Load a saved estimator from disk.

        Args:
            path (os.PathLike | str): Directory produced by `save`.
            map_location (str | torch.device | None): Optional override for torch device.

        Returns:
            CondensiteTorchCDE: Restored estimator ready for inference.

        Raises:
            FileNotFoundError: If expected files are missing.

        Side Effects:
            Reads files from disk and instantiates torch modules.

        Complexity:
            O(params) for loading weights/scalers.
        """
        base = Path(path)
        config_dict = cls._read_json(base / _CONFIG_FILENAME)
        scalers_dict = cls._read_json(base / _SCALERS_FILENAME)
        preprocessor_path = base / _PREPROCESSOR_FILENAME
        preprocessor_data = (
            cls._read_json(preprocessor_path) if preprocessor_path.exists() else None
        )
        metadata = cls._read_json(base / _METADATA_FILENAME)
        config = cls._config_from_dict(config_dict)
        random_seed = int(metadata.get("random_seed", 0))
        estimator = cls(config=config, random_seed=random_seed)
        if map_location is not None:
            estimator._device = torch.device(map_location)
            estimator.config.device = estimator._device
        schema_payload = metadata.get("schema_constraints")
        if schema_payload is not None:
            estimator._schema_constraints = cls._schema_constraints_from_dict(schema_payload)
        elif estimator.config.input_schema is not None:
            estimator._schema_constraints = copy.deepcopy(estimator.config.input_schema)
        estimator._model_card = metadata.get("model_card")
        if estimator._model_card is not None:
            estimator._data_schema = estimator._model_card.get("data_schema")
            estimator._version_info = estimator._model_card.get("versions")
        estimator._set_bandwidths(metadata.get("bandwidths"))
        if preprocessor_data is not None:
            estimator.preprocessor = TabularPreprocessor.from_dict(preprocessor_data)
        else:
            estimator.preprocessor = None
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
        if metadata.get("has_bandwidth_net") and estimator.config.adaptive_bandwidth == "x":
            estimator.bandwidth_net = estimator._build_bandwidth_net(input_dim)
            bw_path = base / _BANDWIDTH_MODEL_FILENAME
            if bw_path.exists():
                bw_state = torch.load(bw_path, map_location=estimator._device)
                estimator.bandwidth_net.load_state_dict(bw_state)
        estimator._best_head_index = int(metadata.get("best_head_index", 0))
        estimator._best_epoch = int(metadata.get("best_epoch", -1))
        restored_epoch = metadata.get("restored_best_epoch")
        estimator._restored_best_epoch = None if restored_epoch is None else int(restored_epoch)
        estimator.training_history.clear()
        estimator._fitted = True
        return estimator

    def _default_y_grid(
        self,
        num_points: int = 128,
        mode: str = "quantile",
    ) -> NDArray[np.float64]:
        mode_literal = cast(GridMode, mode)
        if self._y_train is not None:
            try:
                return make_y_grid(self._y_train, grid_size=num_points, mode=mode_literal)
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
        widths = CondensiteTorchCDE._point_widths_numpy(y_grid)
        increments = pdf * widths[None, :]
        cdf = np.cumsum(increments, axis=1)
        norm = np.clip(cdf[:, -1:], 1e-8, None)
        cdf /= norm
        cdf = np.clip(cdf, 0.0, 1.0)
        cdf[:, -1] = 1.0
        return cdf

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

        result = y_low + rel * widths
        return cast(NDArray[np.float64], result.astype(np.float64, copy=False))

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
            return density_heads[..., 0].astype(np.float64, copy=False)
        resolved = head if head is not None else self.config.bandwidth_strategy
        if isinstance(resolved, str):
            option = resolved.lower()
            if option == "mean":
                averaged = density_heads.mean(axis=-1).astype(np.float64, copy=False)
                return cast(NDArray[np.float64], averaged)
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
        return density_heads[..., index].astype(np.float64, copy=False)

    def _predict_adaptive_bandwidths(self, X_batch: Tensor) -> Tensor | None:
        if self.bandwidth_net is None:
            return None
        raw = self.bandwidth_net(X_batch)
        return F.softplus(raw) + 1e-6

    def _normalize_pdf_heads(self, pdf_heads: Tensor, y_grid: Tensor) -> Tensor:
        widths = self._point_widths_tensor(y_grid).view(1, -1, 1)
        log_widths = torch.log(widths)
        logits = torch.log(torch.clamp(pdf_heads, min=1e-12)) + log_widths
        weights = torch.softmax(logits, dim=1)
        return weights / widths

    def _normalize_pdf_heads_row(self, pdf_row: Tensor, y_grid_row: Tensor) -> Tensor:
        widths = self._point_widths_tensor(y_grid_row)
        log_widths = torch.log(widths).unsqueeze(-1)
        logits = torch.log(torch.clamp(pdf_row, min=1e-12)) + log_widths
        weights = torch.softmax(logits, dim=0)
        return weights / widths.unsqueeze(-1)

    @staticmethod
    def _point_widths_tensor(y_grid: Tensor) -> Tensor:
        if y_grid.ndim != 1 or y_grid.shape[0] < _MIN_GRID_POINTS:
            msg = "y_grid must be 1-D with at least two points."
            raise ValueError(msg)
        diffs = torch.diff(y_grid)
        widths = torch.empty_like(y_grid)
        widths[0] = diffs[0] / 2
        widths[-1] = diffs[-1] / 2
        if y_grid.shape[0] > _MIN_GRID_POINTS:
            widths[1:-1] = (diffs[:-1] + diffs[1:]) / 2
        return torch.clamp(widths, min=1e-8)

    @staticmethod
    def _point_widths_numpy(y_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(y_grid, dtype=np.float64).reshape(-1)
        if arr.ndim != 1 or arr.size < _MIN_GRID_POINTS:
            msg = "y_grid must be 1-D with at least two points."
            raise ValueError(msg)
        diffs = np.diff(arr)
        widths = np.empty_like(arr)
        widths[0] = diffs[0] / 2
        widths[-1] = diffs[-1] / 2
        if arr.size > _MIN_GRID_POINTS:
            widths[1:-1] = (diffs[:-1] + diffs[1:]) / 2
        return np.clip(widths, 1e-12, None)

    def _effective_aux_chunk_size(self) -> int:
        m_aux = max(1, int(self.config.m_aux))
        chunk = self.config.aux_chunk_size
        if chunk is None:
            return min(m_aux, 64)
        if chunk <= 0:
            return m_aux
        return min(m_aux, int(chunk))

    def _inference_row_batch_size(self, total_rows: int) -> int:
        size = self.config.inference_batch_size
        if size is None or size <= 0:
            return total_rows
        return max(1, min(total_rows, int(size)))

    def _inference_grid_chunk_size(self, total_points: int) -> int:
        size = self.config.inference_grid_chunk_size
        if size is None or size <= 0:
            return total_points
        return max(1, min(total_points, int(size)))

    @staticmethod
    def _iter_aux_chunks(
        y_prime: Tensor,
        sample_weights: Tensor,
        chunk_size: int,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        if chunk_size <= 0:
            chunk_size = y_prime.shape[1]
        total = y_prime.shape[1]
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            yield y_prime[:, start:end], sample_weights[:, start:end]

    def _combine_features(self, X_batch: Tensor, y_prime_chunk: Tensor) -> Tensor:
        chunk = y_prime_chunk.shape[1]
        x_expanded = X_batch.unsqueeze(1).expand(-1, chunk, -1)
        y_expanded = y_prime_chunk.unsqueeze(-1)
        features = torch.cat([x_expanded, y_expanded], dim=-1)
        return features.reshape(-1, X_batch.shape[1] + 1)

    def _compute_kernel_targets(
        self,
        y_batch: Tensor,
        y_prime_chunk: Tensor,
        adaptive: Tensor | None,
    ) -> Tensor:
        y_targets = y_batch.unsqueeze(1)
        kernels_per_bandwidth = []
        for idx, bandwidth in enumerate(self._bandwidths):
            if adaptive is None:
                effective_bw: float | Tensor = bandwidth
            else:
                head_scale = adaptive[:, idx].unsqueeze(1)
                effective_bw = head_scale * bandwidth
            kernels_per_bandwidth.append(
                self._kernel_spec.torch_fn(y_targets, y_prime_chunk, effective_bw),
            )
        kernels = torch.stack(kernels_per_bandwidth, dim=-1)
        return kernels.reshape(-1, len(self._bandwidths))

    def _predict_density_row_batch(
        self,
        x_batch: Tensor,
        y_tensor_scaled: Tensor,
        y_tensor: Tensor,
    ) -> Tensor:
        chunk_size = self._inference_grid_chunk_size(y_tensor.shape[0])
        raw_chunks: list[Tensor] = []
        for start in range(0, y_tensor.shape[0], chunk_size):
            end = min(start + chunk_size, y_tensor.shape[0])
            raw_chunks.append(
                self._predict_pdf_chunk(
                    x_batch,
                    y_tensor_scaled[start:end],
                    y_tensor[start:end],
                ),
            )
        raw = (
            torch.cat(raw_chunks, dim=1)
            if raw_chunks
            else torch.zeros(
                (x_batch.shape[0], 0, len(self._bandwidths)),
                device=self._device,
            )
        )
        return self._normalize_pdf_heads(raw, y_tensor)

    def _predict_pdf_chunk(
        self,
        x_batch: Tensor,
        y_scaled_chunk: Tensor,
        y_chunk: Tensor,
    ) -> Tensor:
        if y_scaled_chunk.ndim != 1:
            msg = "y_scaled_chunk must be flattened."
            raise ValueError(msg)
        chunk_size = y_scaled_chunk.shape[0]
        if chunk_size == 0:
            return torch.zeros((x_batch.shape[0], 0, len(self._bandwidths)), device=self._device)
        x_expanded = x_batch.unsqueeze(1).expand(-1, chunk_size, -1)
        y_expanded = y_scaled_chunk.unsqueeze(0).expand(x_batch.shape[0], -1).unsqueeze(-1)
        features = torch.cat([x_expanded, y_expanded], dim=-1).reshape(-1, x_batch.shape[1] + 1)
        model = self.model
        y_scaler = self.y_scaler
        if model is None or y_scaler is None:
            msg = "Model and scaler must be available."
            raise RuntimeError(msg)
        preds = model(features).reshape(x_batch.shape[0], chunk_size, len(self._bandwidths))
        scale = max(float(y_scaler.data_range_), 1e-8)
        return torch.clamp(preds / scale, min=1e-12)

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
        self._importance_sampler = ImportanceSampler.from_array(
            y_scaled.astype(np.float64, copy=False),
        )

    @staticmethod
    def _quantiles_from_cdf(
        cdf: NDArray[np.float64],
        y_grid: NDArray[np.float64],
        probs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        quantiles = np.empty((cdf.shape[0], probs.size), dtype=np.float64)
        for col, prob in enumerate(probs):
            quantiles[:, col] = np.array(
                [np.interp(prob, row, y_grid, left=y_grid[0], right=y_grid[-1]) for row in cdf],
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
        """Capture representative target values for persistence metadata.

        Args:
            None.

        Returns:
            list[float] | None: Quantile summary or `None` if no targets are stored.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(n log n) due to quantile computation.
        """
        if self._y_train is None or self._y_train.size == 0:
            return None
        arr = np.asarray(self._y_train, dtype=np.float64).reshape(-1)
        if arr.size <= _Y_REFERENCE_POINTS:
            return arr.tolist()
        quantiles = np.linspace(0.0, 1.0, _Y_REFERENCE_POINTS)
        summary = np.quantile(arr, quantiles, method="linear")
        return cast(list[float], summary.tolist())

    # ------------------------------------------------------------------ Reproducibility helpers
    def _seed_everything(self) -> None:
        """Seed Python/NumPy/Torch RNGs and enforce deterministic kernels.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.

        Side Effects:
            Sets global RNG seeds and toggles PyTorch deterministic flags.

        Complexity:
            O(1) configuration work.
        """
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        # Deterministic algorithms can be slower but make regression tests trustworthy.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():  # type: ignore[no-untyped-call]
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _environment_versions(self) -> dict[str, Any]:
        """Collect runtime versions for the model card.

        Args:
            None.

        Returns:
            dict[str, Any]: Mapping of package names to version strings.

        Raises:
            None.

        Side Effects:
            Executes a cheap git command (if available) to capture the commit hash.

        Complexity:
            O(1) introspection.
        """
        info: dict[str, Any] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
        cuda_version = torch.version.cuda
        if cuda_version is not None:
            info["cuda"] = cuda_version
        git_commit = self._git_commit()
        if git_commit:
            info["git_commit"] = git_commit
        return info

    @staticmethod
    def _git_commit() -> str | None:
        """Return the current git commit hash if available.

        Args:
            None.

        Returns:
            str | None: Commit hash when running inside a git checkout.

        Raises:
            None.

        Side Effects:
            Invokes `git rev-parse HEAD`.

        Complexity:
            O(1) shell invocation.
        """
        git_exe = shutil.which("git")
        if git_exe is None:
            return None
        try:
            result = subprocess.run(  # noqa: S603
                [git_exe, "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        commit = result.stdout.strip()
        return commit or None

    @staticmethod
    def _summarize_data_schema(
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> dict[str, Any]:
        """Capture lightweight schema stats for the model card.

        Args:
            X (NDArray[np.floating]): Training features used during fit.
            y (NDArray[np.floating]): Target array used during fit.

        Returns:
            dict[str, Any]: Summary with counts, dtypes, and target range.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(n) to compute min/max.
        """
        return {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]) if X.ndim == _GRID_DIMENSION else 0,
            "x_dtype": str(X.dtype),
            "y_dtype": str(y.dtype),
            "y_min": float(np.min(y)) if y.size else 0.0,
            "y_max": float(np.max(y)) if y.size else 0.0,
        }

    def _schema_from_training(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> SchemaConstraints | None:
        """Build schema constraints using fitted preprocessors and target range."""
        if X.ndim != _GRID_DIMENSION:
            return None
        preprocessor = self.preprocessor
        numeric_indices: list[int] | None = None
        categorical_indices: list[int] | None = None
        if preprocessor is not None and preprocessor.numeric_indices:
            numeric_indices = list(preprocessor.numeric_indices)
        if preprocessor is not None and preprocessor.categorical_indices:
            categorical_indices = list(preprocessor.categorical_indices)
        max_card = None
        if preprocessor is not None and preprocessor.categorical_indices:
            counts = [
                len(preprocessor.categorical_categories_.get(idx, []))
                for idx in preprocessor.categorical_indices
            ]
            if counts:
                max_card = max(counts)
        y_min = float(np.min(y)) if y.size else None
        y_max = float(np.max(y)) if y.size else None
        return SchemaConstraints(
            numeric_indices=numeric_indices,
            categorical_indices=categorical_indices,
            max_categorical_cardinality=max_card,
            allow_missing_numeric=True,
            allow_missing_categorical=True,
            y_min=y_min,
            y_max=y_max,
        )

    def _config_hash(self) -> str:
        """Compute a stable hash of the serialized config.

        Args:
            None.

        Returns:
            str: Hex digest identifying the config payload.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(len(config_json)).
        """
        payload = json.dumps(self._config_to_dict(self.config), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _schema_for_validation(self, *, include_targets: bool) -> SchemaConstraints | None:
        """Return a defensive copy of the schema constraints for validation."""
        schema = self._schema_constraints
        if schema is None:
            return None
        return SchemaConstraints(
            numeric_indices=(
                list(schema.numeric_indices) if schema.numeric_indices is not None else None
            ),
            categorical_indices=(
                list(schema.categorical_indices) if schema.categorical_indices is not None else None
            ),
            max_categorical_cardinality=schema.max_categorical_cardinality,
            allow_missing_numeric=schema.allow_missing_numeric,
            allow_missing_categorical=schema.allow_missing_categorical,
            y_min=schema.y_min if include_targets else None,
            y_max=schema.y_max if include_targets else None,
        )

    @staticmethod
    def _merge_schema_constraints(
        base: SchemaConstraints | None,
        derived: SchemaConstraints | None,
    ) -> SchemaConstraints | None:
        """Combine base/user schema with derived schema information."""
        if base is None and derived is None:
            return None
        if base is None:
            return copy.deepcopy(derived)
        if derived is None:
            return copy.deepcopy(base)

        def _pick_seq(
            primary: Sequence[int] | None,
            secondary: Sequence[int] | None,
        ) -> list[int] | None:
            if primary:
                return list(primary)
            if secondary:
                return list(secondary)
            return None

        def _pick_card(primary: int | None, secondary: int | None) -> int | None:
            candidates = [val for val in (primary, secondary) if val is not None]
            return min(candidates) if candidates else None

        return SchemaConstraints(
            numeric_indices=_pick_seq(base.numeric_indices, derived.numeric_indices),
            categorical_indices=_pick_seq(base.categorical_indices, derived.categorical_indices),
            max_categorical_cardinality=_pick_card(
                base.max_categorical_cardinality,
                derived.max_categorical_cardinality,
            ),
            allow_missing_numeric=base.allow_missing_numeric and derived.allow_missing_numeric,
            allow_missing_categorical=base.allow_missing_categorical
            and derived.allow_missing_categorical,
            y_min=base.y_min if base.y_min is not None else derived.y_min,
            y_max=base.y_max if base.y_max is not None else derived.y_max,
        )

    @staticmethod
    def _schema_constraints_to_dict(schema: SchemaConstraints) -> dict[str, Any]:
        """Serialize schema constraints to JSON-safe payload."""
        return {
            "numeric_indices": (
                list(schema.numeric_indices) if schema.numeric_indices is not None else None
            ),
            "categorical_indices": (
                list(schema.categorical_indices) if schema.categorical_indices is not None else None
            ),
            "max_categorical_cardinality": schema.max_categorical_cardinality,
            "allow_missing_numeric": schema.allow_missing_numeric,
            "allow_missing_categorical": schema.allow_missing_categorical,
            "y_min": schema.y_min,
            "y_max": schema.y_max,
        }

    @staticmethod
    def _schema_constraints_from_dict(data: Mapping[str, Any]) -> SchemaConstraints:
        """Deserialize schema constraints from JSON payload."""
        return SchemaConstraints(
            numeric_indices=data.get("numeric_indices"),
            categorical_indices=data.get("categorical_indices"),
            max_categorical_cardinality=data.get("max_categorical_cardinality"),
            allow_missing_numeric=bool(data.get("allow_missing_numeric", True)),
            allow_missing_categorical=bool(data.get("allow_missing_categorical", True)),
            y_min=data.get("y_min"),
            y_max=data.get("y_max"),
        )

    def _build_model_card(
        self,
        *,
        schema: dict[str, Any] | None,
        training_time_seconds: float,
        best_metric: float | None,
        monitor_metric: str,
    ) -> dict[str, Any]:
        """Finalize the model card payload after training completes.

        Args:
            schema (dict[str, Any] | None): Data schema summary collected at fit time.
            training_time_seconds (float): Wall-clock time spent training.
            best_metric (float | None): Best monitor metric observed.
            monitor_metric (str): Name of the monitored validation metric.

        Returns:
            dict[str, Any]: JSON-serializable metadata describing the run.

        Raises:
            None.

        Side Effects:
            None.

        Complexity:
            O(1).
        """
        schema_payload = dict(schema or {})
        if self._schema_constraints is not None:
            schema_payload["constraints"] = self._schema_constraints_to_dict(
                self._schema_constraints,
            )
        versions = self._version_info or self._environment_versions()
        early_stopping_triggered = self._restored_best_epoch is not None
        return {
            "config": self._config_to_dict(self.config),
            "config_hash": self._config_hash(),
            "random_seed": self.random_seed,
            "reproducible": bool(self.config.reproducible),
            "versions": versions,
            "data_schema": schema_payload,
            "training": {
                "time_seconds": float(training_time_seconds),
                "monitor_metric": monitor_metric,
                "best_metric": best_metric,
                "epochs_requested": int(self.config.epochs),
                "epochs_ran": int(self._current_epoch + 1),
                "best_epoch": int(self._best_epoch),
                "restored_epoch": (
                    None if self._restored_best_epoch is None else int(self._restored_best_epoch)
                ),
                "early_stopping_triggered": early_stopping_triggered,
                "sampler_calls": int(self._sampler_call),
                "m_aux": int(self.config.m_aux),
                "sampler": self.config.sampler,
            },
        }

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

    def _preprocessor_to_dict(self) -> dict[str, Any] | None:
        if self.preprocessor is None:
            return None
        return self.preprocessor.to_dict()

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

    def _fit_preprocessor(self, X: NDArray[np.object_]) -> NDArray[np.float64]:
        if self.preprocessor is not None:
            features = self.preprocessor.fit_transform(X)
        else:
            features = np.asarray(X, dtype=np.float64)
        self.x_scaler = StandardScaler().fit(features)
        transformed = self.x_scaler.transform(features)
        return cast(NDArray[np.float64], transformed)

    def _transform_with_preprocessor(self, X: NDArray[np.object_]) -> NDArray[np.float64]:
        assert self.x_scaler is not None
        if self.preprocessor is not None:
            features = self.preprocessor.transform(X)
        else:
            features = np.asarray(X, dtype=np.float64)
        transformed = self.x_scaler.transform(features)
        return cast(NDArray[np.float64], transformed)

    def _build_local_grids(
        self,
        X: NDArray[np.object_],
        params: dict[str, float | int] | None,
    ) -> NDArray[np.float64]:
        config = params or {}
        grid_size = int(config.get("grid_size", 64))
        q_low = float(config.get("q_low", 0.01))
        q_high = float(config.get("q_high", 0.99))
        padding = float(config.get("padding", 0.1))
        head_raw = config.get("head")
        if head_raw is None or isinstance(head_raw, (int, str)):
            head = head_raw
        else:
            msg = "head parameter must be int, str, or None."
            raise TypeError(msg)
        reference = config.get("reference_grid")
        y_grid_np = None
        if reference is not None:
            y_grid_np = np.asarray(reference, dtype=np.float64)
        cache_key = self._local_grid_cache_key(
            X,
            grid_size=grid_size,
            q_low=q_low,
            q_high=q_high,
            padding=padding,
            head=head,
            reference_grid=y_grid_np,
        )
        cached = self._local_grid_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        grids = local_grids.make_local_grid(
            self,
            X,
            grid_size=grid_size,
            q_low=q_low,
            q_high=q_high,
            padding=padding,
            y_grid=y_grid_np,
            head=head,
        )
        self._store_local_grid_cache(cache_key, grids)
        return grids

    def _local_grid_cache_key(  # noqa: PLR0913
        self,
        X: NDArray[np.object_],
        *,
        grid_size: int,
        q_low: float,
        q_high: float,
        padding: float,
        head: int | str | None,
        reference_grid: NDArray[np.float64] | None,
    ) -> tuple[str, int, float, float, float, str | None, str | None]:
        features = self._transform_with_preprocessor(X).astype(np.float32, copy=False)
        features_bytes = np.ascontiguousarray(features).tobytes()
        feature_digest = hashlib.sha256(features_bytes).hexdigest()
        reference_digest = None
        if reference_grid is not None:
            reference_arr = np.asarray(reference_grid, dtype=np.float64)
            reference_bytes = np.ascontiguousarray(reference_arr).tobytes()
            reference_digest = hashlib.sha256(reference_bytes).hexdigest()
        head_key = None if head is None else str(head)
        return (
            feature_digest,
            int(grid_size),
            float(q_low),
            float(q_high),
            float(padding),
            head_key,
            reference_digest,
        )

    def _store_local_grid_cache(
        self,
        key: tuple[str, int, float, float, float, str | None, str | None],
        grids: NDArray[np.float64],
    ) -> None:
        self._local_grid_cache[key] = grids.astype(np.float64, copy=True)
        self._local_grid_cache.move_to_end(key)
        while len(self._local_grid_cache) > _LOCAL_GRID_CACHE_SIZE:
            self._local_grid_cache.popitem(last=False)

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
            return cast(dict[str, Any], json.load(handle))

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

    def _process_y_grid(
        self,
        X: NDArray[np.object_],
        y_grid: NDArray[np.floating],
    ) -> tuple[NDArray[np.float64], bool]:
        arr = np.asarray(y_grid, dtype=np.float64)
        if arr.ndim == 1:
            return self._validate_y_grid(arr), False
        if arr.ndim != _GRID_DIMENSION:
            msg = "y_grid must be 1-D or 2-D."
            raise ValueError(msg)
        if arr.shape[0] != X.shape[0]:
            msg = f"y_grid with shape {arr.shape} must match X rows ({X.shape[0]})."
            raise ValueError(msg)
        if np.any(np.diff(arr, axis=1) <= 0):
            msg = "All per-row grids must be strictly increasing."
            raise ValueError(msg)
        return arr.astype(np.float64, copy=False), True

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


class _NormalizationAccumulator:
    def __init__(self, batch_size: int, num_heads: int, device: torch.device) -> None:
        self._weighted_sum = torch.zeros((batch_size, num_heads), device=device)
        self._weight_sum = torch.zeros((batch_size, 1), device=device)

    def accumulate(self, predictions: Tensor, weights: Tensor) -> None:
        self._weighted_sum += torch.sum(predictions * weights, dim=1)
        self._weight_sum += torch.sum(weights, dim=1)

    def penalty(self) -> Tensor:
        integrals = self._weighted_sum / torch.clamp(self._weight_sum, min=1e-6)
        penalty = (integrals - 1.0) ** 2
        return cast(Tensor, penalty.mean())


__all__: tuple[str, ...] = ("CondensiteTorchCDE", "CondensiteTorchCDEConfig")
