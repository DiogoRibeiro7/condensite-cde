"""Tests for tuning utilities and validation-driven early stopping."""

from __future__ import annotations

import importlib

import numpy as np

from condensite_cde.tune import tune_bandwidth_m_aux


def _make_small_dataset(n_samples: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 2))
    noise = (0.2 + 0.1 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.5 * X[:, 0] - 0.25 * X[:, 1] + noise
    return X, y


def test_training_history_contains_validation_metrics(torch_available: bool) -> None:
    assert torch_available
    module = importlib.import_module("condensite_torch")
    CondensiteTorchCDE = module.CondensiteTorchCDE
    CondensiteTorchCDEConfig = module.CondensiteTorchCDEConfig
    X, y = _make_small_dataset()
    config = CondensiteTorchCDEConfig(
        epochs=4,
        patience=1,
        batch_size=32,
        m_aux=16,
        bandwidth=0.15,
        sampler="stratified",
        val_fraction=0.2,
        monitor_metric="val_nll",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=2).fit(X, y)
    assert estimator.training_history, "Expected non-empty history"
    assert len(estimator.training_history) <= config.epochs
    for record in estimator.training_history:
        assert "val_crps" in record
        assert "val_nll" in record
        assert "val_integral_error" in record


def test_tuner_returns_best_configuration(torch_available: bool, tmp_path) -> None:
    assert torch_available
    module = importlib.import_module("condensite_torch")
    CondensiteTorchCDEConfig = module.CondensiteTorchCDEConfig
    X, y = _make_small_dataset()
    base_config = CondensiteTorchCDEConfig(
        epochs=3,
        patience=1,
        batch_size=32,
        val_fraction=0.2,
        sampler="iid",
    )
    bandwidths = [0.08, 0.12]
    aux_values = [12, 16]
    result = tune_bandwidth_m_aux(
        X,
        y,
        bandwidths=bandwidths,
        m_aux_values=aux_values,
        base_config=base_config,
        metric="val_crps",
        random_seed=10,
        run_root=tmp_path,
    )
    assert result.best_config.bandwidth in {0.08, 0.12}
    assert result.best_config.m_aux in {12, 16}
    assert len(result.history) == len(bandwidths) * len(aux_values)
    assert result.metric_name == "val_crps"
    json_artifacts = list(result.run_dir.rglob("*.json"))
    assert json_artifacts
    assert all(path.read_text(encoding="utf-8").endswith("\n") for path in json_artifacts)


def test_tuner_uses_cache_for_identical_configs(
    torch_available: bool,
    monkeypatch,
    tmp_path,
) -> None:
    assert torch_available
    module = importlib.import_module("condensite_torch")
    CondensiteTorchCDE = module.CondensiteTorchCDE
    CondensiteTorchCDEConfig = module.CondensiteTorchCDEConfig
    X, y = _make_small_dataset()
    base_config = CondensiteTorchCDEConfig(
        epochs=2,
        patience=1,
        batch_size=32,
        val_fraction=0.2,
    )
    call_count = 0
    original_fit = CondensiteTorchCDE.fit

    def _counting_fit(self, X_train, y_train):
        nonlocal call_count
        call_count += 1
        return original_fit(self, X_train, y_train)

    monkeypatch.setattr(CondensiteTorchCDE, "fit", _counting_fit)
    result = tune_bandwidth_m_aux(
        X,
        y,
        bandwidths=[0.1, 0.1],
        m_aux_values=[16],
        base_config=base_config,
        run_root=tmp_path,
    )
    assert call_count == 1
    assert len(result.history) == 1
    assert result.run_dir.exists()


def test_tuner_resume_skips_completed_trials(
    torch_available: bool,
    monkeypatch,
    tmp_path,
) -> None:
    assert torch_available
    module = importlib.import_module("condensite_torch")
    CondensiteTorchCDE = module.CondensiteTorchCDE
    CondensiteTorchCDEConfig = module.CondensiteTorchCDEConfig
    X, y = _make_small_dataset()
    base_config = CondensiteTorchCDEConfig(
        epochs=2,
        patience=1,
        batch_size=32,
        val_fraction=0.2,
    )
    run_name = "resume-run"
    initial = tune_bandwidth_m_aux(
        X,
        y,
        bandwidths=[0.12],
        m_aux_values=[12],
        base_config=base_config,
        run_root=tmp_path,
        run_name=run_name,
    )
    assert initial.history
    config_path = tmp_path / run_name / "config.json"
    assert config_path.exists()

    def _fail_fit(self, *_args, **_kwargs):
        raise AssertionError("fit should not be invoked during resume")

    monkeypatch.setattr(CondensiteTorchCDE, "fit", _fail_fit)
    resumed = tune_bandwidth_m_aux(
        X,
        y,
        bandwidths=[0.12],
        m_aux_values=[12],
        base_config=base_config,
        run_root=tmp_path,
        run_name=run_name,
        resume=True,
    )
    assert resumed.history == initial.history
