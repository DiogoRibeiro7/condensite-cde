from __future__ import annotations

from pathlib import Path


def replace(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Missing expected text in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


path = "src/condensite_torch/estimator.py"
replace(
    path,
    "        self._local_grid_cache.clear()\n"
    "        self._set_bandwidths()\n"
    "        self._y_train = y_arr.copy()\n"
    "        self._data_schema = self._summarize_data_schema(X_arr, y_arr)\n"
    "        self._version_info = self._environment_versions()\n"
    "        start_time = time.perf_counter()\n",
    "        self._local_grid_cache.clear()\n"
    "        self._set_bandwidths()\n"
    "        self._version_info = self._environment_versions()\n"
    "        start_time = time.perf_counter()\n",
)
replace(
    path,
    "        if self.config.preprocessor is not None:\n"
    "            self.preprocessor = TabularPreprocessor(self.config.preprocessor)\n"
    "        else:\n"
    "            self.preprocessor = None\n"
    "        X_features = self._fit_preprocessor(X_arr)\n"
    "        self.y_scaler = MinMaxScaler1D().fit(y_arr)\n"
    "        derived_schema = self._schema_from_training(X_arr, y_arr)\n"
    "        self._schema_constraints = self._merge_schema_constraints(\n"
    "            self._schema_constraints,\n"
    "            derived_schema,\n"
    "        )\n"
    "\n"
    "        X_scaled: NDArray[np.float32] = X_features.astype(np.float32)\n"
    "        y_scaled: NDArray[np.float32] = self.y_scaler.transform(y_arr)\n"
    "\n"
    "        val_eval_data: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None\n"
    "        if X_val is None or y_val is None:\n"
    "            X_train_scaled, y_train_scaled, val_idx = self._train_val_split(X_scaled, y_scaled)\n"
    "            if val_idx is not None and val_idx.size > 0:\n"
    "                val_eval_data = (X_arr[val_idx], y_arr[val_idx])\n"
    "        else:\n"
    "            X_val_arr = np.asarray(X_val, dtype=object)\n"
    "            y_val_arr = np.asarray(y_val, dtype=np.float64).reshape(-1)\n"
    "            if X_val_arr.shape[0] != y_val_arr.shape[0]:\n"
    "                msg = \"Validation X and y must match in the first dimension.\"\n"
    "                raise ValueError(msg)\n"
    "            self._transform_with_preprocessor(X_val_arr)  # Validate shape\n"
    "            self.y_scaler.transform(y_val_arr)\n"
    "            X_train_scaled, y_train_scaled = X_scaled, y_scaled\n"
    "            if X_val_arr.size > 0:\n"
    "                val_eval_data = (X_val_arr, y_val_arr)\n"
    "            else:\n"
    "                val_eval_data = None\n",
    "        val_eval_data: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None\n"
    "        X_train_arr = X_arr\n"
    "        y_train_arr = y_arr\n"
    "        if X_val is None or y_val is None:\n"
    "            train_idx, val_idx = self._train_val_indices(X_arr.shape[0])\n"
    "            if val_idx is not None and val_idx.size > 0:\n"
    "                X_train_arr = X_arr[train_idx]\n"
    "                y_train_arr = y_arr[train_idx]\n"
    "                val_eval_data = (X_arr[val_idx], y_arr[val_idx])\n"
    "        else:\n"
    "            X_val_arr = np.asarray(X_val, dtype=object)\n"
    "            y_val_arr = np.asarray(y_val, dtype=np.float64).reshape(-1)\n"
    "            if X_val_arr.shape[0] != y_val_arr.shape[0]:\n"
    "                msg = \"Validation X and y must match in the first dimension.\"\n"
    "                raise ValueError(msg)\n"
    "            if X_val_arr.size > 0:\n"
    "                val_eval_data = (X_val_arr, y_val_arr)\n"
    "\n"
    "        self._y_train = y_train_arr.copy()\n"
    "        self._data_schema = self._summarize_data_schema(X_train_arr, y_train_arr)\n"
    "        if self.config.preprocessor is not None:\n"
    "            self.preprocessor = TabularPreprocessor(self.config.preprocessor)\n"
    "        else:\n"
    "            self.preprocessor = None\n"
    "        X_train_scaled = self._fit_preprocessor(X_train_arr).astype(np.float32)\n"
    "        self.y_scaler = MinMaxScaler1D().fit(y_train_arr)\n"
    "        y_train_scaled = self.y_scaler.transform(y_train_arr)\n"
    "        derived_schema = self._schema_from_training(X_train_arr, y_train_arr)\n"
    "        self._schema_constraints = self._merge_schema_constraints(\n"
    "            self._schema_constraints,\n"
    "            derived_schema,\n"
    "        )\n"
    "        if val_eval_data is not None:\n"
    "            X_val_eval, y_val_eval = val_eval_data\n"
    "            self._transform_with_preprocessor(X_val_eval)\n"
    "            self.y_scaler.transform(y_val_eval)\n",
)
replace(
    path,
    "        x_dim = X_scaled.shape[1]\n",
    "        x_dim = X_train_scaled.shape[1]\n",
)
replace(
    path,
    "    def _train_val_split(\n"
    "        self,\n"
    "        X: NDArray[np.float32],\n"
    "        y: NDArray[np.float32],\n"
    "    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64] | None]:\n"
    "        if self.config.val_fraction <= 0.0 or X.shape[0] <= 1:\n"
    "            return X, y, None\n"
    "        n_samples = X.shape[0]\n"
    "        n_val = max(1, int(n_samples * self.config.val_fraction))\n"
    "        if n_val >= n_samples:\n"
    "            n_val = n_samples - 1\n"
    "        indices = np.arange(n_samples)\n"
    "        self._rng.shuffle(indices)\n"
    "        val_idx = indices[:n_val]\n"
    "        train_idx = indices[n_val:]\n"
    "        return X[train_idx], y[train_idx], val_idx\n",
    "    def _train_val_indices(\n"
    "        self,\n"
    "        n_samples: int,\n"
    "    ) -> tuple[NDArray[np.int64], NDArray[np.int64] | None]:\n"
    "        indices = np.arange(n_samples, dtype=np.int64)\n"
    "        if self.config.val_fraction <= 0.0 or n_samples <= 1:\n"
    "            return indices, None\n"
    "        n_val = max(1, int(n_samples * self.config.val_fraction))\n"
    "        if n_val >= n_samples:\n"
    "            n_val = n_samples - 1\n"
    "        self._rng.shuffle(indices)\n"
    "        val_idx = indices[:n_val]\n"
    "        train_idx = indices[n_val:]\n"
    "        return train_idx, val_idx\n",
)
replace(
    path,
    "        quantiles = self._quantiles_from_cdf(\n"
    "            cdf,\n"
    "            grid_arr,\n"
    "            np.array([alpha_value], dtype=np.float64),\n"
    "        )\n",
    "        quantile_probability = alpha_value if side_norm == \"right\" else 1.0 - alpha_value\n"
    "        quantiles = self._quantiles_from_cdf(\n"
    "            cdf,\n"
    "            grid_arr,\n"
    "            np.array([quantile_probability], dtype=np.float64),\n"
    "        )\n",
)

# Extend existing reviewer regression file with leakage coverage.
test_path = Path("tests/unit/test_reviewer_estimator_fixes.py")
test_text = test_path.read_text(encoding="utf-8")
addition = '''\n\ndef test_internal_validation_split_fits_scalers_on_training_rows_only() -> None:\n    X = np.arange(20, dtype=np.float64).reshape(10, 2)\n    y = np.linspace(0.0, 9.0, 10)\n    config = CondensiteTorchCDEConfig(\n        hidden_sizes=(4,),\n        epochs=1,\n        patience=1,\n        batch_size=4,\n        m_aux=4,\n        val_fraction=0.2,\n    )\n    estimator = CondensiteTorchCDE(config=config, random_seed=3)\n    probe = CondensiteTorchCDE(config=config, random_seed=3)\n    train_idx, val_idx = probe._train_val_indices(len(y))\n    assert val_idx is not None\n\n    estimator.fit(X, y)\n\n    assert estimator.y_scaler is not None\n    assert estimator._y_train is not None\n    assert estimator.y_scaler.min_ == pytest.approx(float(y[train_idx].min()))\n    assert estimator.y_scaler.max_ == pytest.approx(float(y[train_idx].max()))\n    np.testing.assert_array_equal(np.sort(estimator._y_train), np.sort(y[train_idx]))\n    assert not np.any(np.isin(y[val_idx], estimator._y_train))\n'''
if "test_internal_validation_split_fits_scalers_on_training_rows_only" not in test_text:
    test_path.write_text(test_text + addition, encoding="utf-8")

# Add left-tail expected-shortfall regression alongside the existing right-tail test.
tail_path = Path("tests/test_condensite_cde.py")
tail_text = tail_path.read_text(encoding="utf-8")
marker = "def test_expected_shortfall_exceeds_quantile(trained_estimator) -> None:\n"
if marker not in tail_text:
    raise RuntimeError("Expected right-tail ES test not found")
left_test = '''\n\ndef test_left_expected_shortfall_is_below_lower_tail_quantile(trained_estimator) -> None:\n    estimator, X, _y, grid = trained_estimator\n    alpha = 0.9\n    lower_quantile = estimator.predict_quantile(X[:6], 1.0 - alpha, y_grid=grid)\n    es_values = estimator.expected_shortfall(X[:6], alpha=alpha, side="left", y_grid=grid)\n    assert lower_quantile.shape == es_values.shape == (6,)\n    assert np.all(es_values <= lower_quantile + 1e-6)\n\n'''
if "test_left_expected_shortfall_is_below_lower_tail_quantile" not in tail_text:
    tail_path.write_text(tail_text.replace(marker, left_test + marker, 1), encoding="utf-8")
