# condensite-cde

PyTorch tabular conditional density estimator utilities with a clean developer experience.

## Features

- Modern `src/`-layout with Poetry management (`>=3.10,<3.13`)
- Ruff and MyPy in strict mode keep quality high
- Ready-to-go CI pipeline plus GitHub templates and governance docs

## Quickstart

```python
import numpy as np
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

X = np.random.normal(size=(256, 3))
y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * np.random.normal(size=X.shape[0])
grid = np.linspace(y.min() - 1.0, y.max() + 1.0, 96)

config = CondensiteTorchCDEConfig(m_aux=64, bandwidth=0.12, epochs=8, patience=2)
estimator = CondensiteTorchCDE(config=config, random_seed=7).fit(X, y)
pdf = estimator.predict_density(X[:2], grid)

estimator.save("artifacts/basic")
restored = CondensiteTorchCDE.load("artifacts/basic", map_location="cpu")
assert np.allclose(pdf, restored.predict_density(X[:2], grid), atol=1e-5)
```

Use `examples/basic_tabular.py` for a fuller walkthrough including validation metrics.

## What Is Condensite?

1. Condensite models the conditional density `p(y|x)` directly for tabular data.
2. It augments each minibatch with auxiliary `y'` points sampled across the target range.
3. A Gaussian kernel with bandwidth `h` defines the density target for every `y'`.
4. The model learns to map `[x, y']` to `K_h(y - y')`, enabling smooth PDFs and CDFs.
5. Streaming `y'` keeps memory usage flat, no need to store expanded datasets.
6. Stratified or Sobol sampling reduces variance relative to IID auxiliaries.
7. Positive MLP heads (softplus) remove the need for clamping densities post-hoc.
8. Prediction reuses the same head over a user-defined `y_grid`.
9. CDFs arise by integrating predicted PDFs, enabling quantiles or inverse-CDF sampling.
10. Metrics such as CRPS/NLL guide hyper-parameters like `m_aux`, bandwidth, or sampling.

## Tuning Cheatsheet

- **Auxiliary count (`m_aux`)**: Use `64..256` for most problems; start small (64) for prototyping and scale up if densities look jagged.
- **Bandwidth (`h`)**: Work in the scaled target space; `0.05..0.15` usually balances smoothness vs. fidelity. Sweep using `examples/tuning_bandwidth.py`.
- **`y_grid` selection**: Provide a quantile-spaced grid (e.g., percentiles of the training target) to focus resolution where data mass lives; supplement with min/max padding.
- **Sampling strategy**: Sobol QMC (`sampler="sobol"`) is the recommended baseline, followed by stratified sampling; `examples/compare_aux_sampling.py` demonstrates the trade-offs. Use `sampler="importance"` to reweight y' draws via the empirical target histogram for better tail coverage.
- **Auto tuning**: Run `condensite_cde.tune.tune_bandwidth_m_aux` (see `examples/tune_bandwidth.py`) to grid-search bandwidths and auxiliary counts using validation CRPS/NLL.
- **Multi-bandwidth heads**: Set `bandwidths=[0.06, 0.12, 0.2]` to train one head per bandwidth; use `bandwidth_strategy="mean"` (or pass `head="mean"`, `head="best"`, or a head index when calling `predict_*`) to select how densities are combined at inference. `head="best"` reuses the validation metric (CRPS/NLL) to pick the sharpest head at inference.
- **Adaptive bandwidths**: Switch on `adaptive_bandwidth="x"` to predict positive per-sample bandwidth scalings that modulate smoothing automatically while keeping compatibility with fixed-bandwidth training.
- **Normalization penalty**: Tune `normalization_lambda>0` to add a differentiable squared-integral penalty so the raw heads stay close to valid PDFs even before post-hoc renormalization.
- **Sampling benchmark**: `scripts/aux_sampling_benchmark.py` trains a small model with `sampler in {iid, sobol, stratified, importance}` and prints a JSON summary of CRPS/NLL so you can quantify the trade-offs.
- **Calibration diagnostics**: `scripts/calibration_report.py` emits PIT histograms and coverage stats so you can monitor probabilistic calibration over time.
- **Split-conformal intervals**: Wrap the estimator with `ConformalCDEWrapper` to obtain finite-sample predictive intervals; see `examples/conformal_intervals.py`.
- **AMP & GPU**: Set `amp=True` when training on CUDA devices; automatic casting and gradient scaling are enabled through PyTorch AMP.

## Automated Tuning

```python
from condensite_cde.tune import tune_bandwidth_m_aux
from condensite_torch import CondensiteTorchCDEConfig

result = tune_bandwidth_m_aux(
    X_train,
    y_train,
    bandwidths=[0.08, 0.12, 0.16],
    m_aux_values=[64, 96],
    base_config=CondensiteTorchCDEConfig(epochs=6, patience=2, val_fraction=0.2),
    metric="val_crps",
)
best_config = result.best_config
print(f"Best bandwidth={best_config.bandwidth}, m_aux={best_config.m_aux}")
```

The tuner logs every trial (bandwidth, auxiliary count, validation metric) so you can rank candidates without writing custom loops.

## Save & Load

- `estimator.save(path)` writes `model.pt`, `config.json`, `scalers.json`, and metadata (including quantile summaries) into the target directory.
- `CondensiteTorchCDE.load(path, map_location="cpu")` restores the estimator, scalers, and configuration for immediate inference or continued training.
- The round-trip is covered by unit tests to guarantee deterministic density predictions after reloads.

## Examples

Run the scripts with Poetry to explore practical settings:

```bash
poetry run python examples/basic_tabular.py
poetry run python examples/tuning_bandwidth.py
poetry run python examples/compare_aux_sampling.py
poetry run python examples/tune_bandwidth.py
poetry run python examples/decision_metrics.py
poetry run python examples/conformal_intervals.py
poetry run python scripts/aux_sampling_benchmark.py > benchmark.json
poetry run python scripts/calibration_report.py
poetry run python benchmarks/run_all.py
```

## Getting Started

```bash
poetry install
poetry run pytest -q
poetry run ruff check .
poetry run mypy src
```

## Development Workflow

1. Create a feature branch.
2. Run `poetry run ruff check .` and `poetry run mypy src` before committing.
3. Add or update tests under `tests/` and run `poetry run pytest`.
4. Open a pull request using the provided template.

## Licensing

This project is licensed under the [MIT License](LICENSE).
