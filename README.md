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
quantiles = estimator.predict_quantile(X[:2], [0.1, 0.5, 0.9], y_grid=grid)
interval_lo, interval_hi = estimator.predict_interval(X[:2], coverage=0.9, y_grid=grid)

estimator.save("artifacts/basic")
restored = CondensiteTorchCDE.load("artifacts/basic", map_location="cpu")
assert np.allclose(pdf, restored.predict_density(X[:2], grid), atol=1e-5)
```

Use `examples/basic_tabular.py` for a fuller walkthrough including validation metrics, `examples/persistence_roundtrip.py` to see save/load + evaluation helpers in action, `examples/quantiles_and_intervals.py` for quantiles + predictive intervals, and `examples/tail_risk.py` for tail probability / expected shortfall APIs.

## Common Pitfalls

- **Missing validation split**: set `val_fraction` or pass `(X_val, y_val)` when you want early stopping or `head="best"` selection; otherwise validation metrics stay `None`.
- **Grid coverage**: always provide a `y_grid` that spans your target distribution; forgetting to pad min/max leads to clipped densities.
- **Baseline refresh**: after making deterministic changes to density calculation, run `poetry run python scripts/generate_regression_baselines.py` before committing so regression snapshots stay in sync.

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
- **Sampling strategy**: Sobol QMC (`sampler="sobol"`) is the recommended baseline (and default). Latin hypercube (`sampler="lhs"`) keeps one sample per bin with randomized permutations, stratified sampling ensures fixed bins, and `sampler="fixed_grid"` reuses deterministic linspace points for debugging. `examples/compare_aux_sampling.py` covers the trade-offs, and `sampler="importance"` reweights y′ draws via the empirical target histogram for better tail coverage.
- **Auto tuning**: Run `condensite_cde.tune.tune_bandwidth_m_aux` (see `examples/tune_bandwidth.py`) to grid-search bandwidths and auxiliary counts using validation CRPS/NLL.
- **Multi-bandwidth heads**: Set `bandwidths=[0.06, 0.12, 0.2]` to train one head per bandwidth; use `bandwidth_strategy="mean"` (or pass `head="mean"`, `head="best"`, or a head index when calling `predict_*`) to select how densities are combined at inference. `head="best"` reuses the validation metric (CRPS/NLL) to pick the sharpest head at inference.
- **Adaptive bandwidths**: Switch on `adaptive_bandwidth="x"` to predict positive per-sample bandwidth scalings that modulate smoothing automatically while keeping compatibility with fixed-bandwidth training.
- **Normalization penalty**: Tune `normalization_lambda>0` to add a differentiable squared-integral penalty so the raw heads stay close to valid PDFs even before post-hoc renormalization.
- **Pluggable kernels/losses**: Choose `kernel in {"gaussian","epanechnikov"}` plus `loss in {"mse","mae"}` (e.g., `CondensiteTorchCDEConfig(kernel="epanechnikov", loss="mae")`) to explore alternate smoothing/optimization objectives without forking the trainer—see `examples/custom_kernel_loss.py`.
- **Reproducible mode + model cards**: Set `CondensiteTorchCDEConfig(reproducible=True)` to seed Python/NumPy/Torch, pin deterministic kernels, and emit a model card (`estimator.model_card()`) capturing config hashes, schema stats, runtime versions, and training/early-stopping summaries. The same metadata is persisted in `metadata.json` when you call `save()`.
- **Evaluation helper**: Call `estimator.evaluate(X, y)` to obtain CRPS/NLL/integral-error diagnostics without writing boilerplate loops; `examples/persistence_roundtrip.py` shows how to pair it with save/load.
- **Quantiles & tail risk**: `predict_quantile`, `predict_interval`, `predict_tail_prob`, and `expected_shortfall` expose decision metrics; see `examples/quantiles_and_intervals.py` and `examples/tail_risk.py`.
- **Input validation**: Pass `CondensiteTorchCDEConfig(input_schema=SchemaConstraints(...))` to enforce dtype/missingness/cardinality/target bounds up front—the estimator calls `validate_inputs` automatically during `fit`/`predict` so bad batches fail fast with actionable errors.
- **Fast inference**: Control memory/throughput by setting `inference_batch_size` (rows) and `inference_grid_chunk_size` (grid points) so `predict_density` streams large evaluations without blowing up RAM. Run `scripts/inference_benchmark.py` to see the impact on a medium dataset.
- **Permutation importance**: `condensite_torch.permutation_importance` perturbs features and recomputes CRPS/NLL to quantify their impact; `examples/permutation_importance.py` prints mean/std importances for a toy dataset.
- **What-if analysis**: `condensite_torch.what_if` mutates selected features and reports how quantiles/tails/pdf/cdf shift; `examples/what_if.py` shows a minimal counterfactual report.
- **Tabular preprocessing**: Mixed numeric/categorical inputs are handled automatically via `TabularPreprocessorConfig` (median/mode imputation, optional missing indicators, one-hot with deterministic ordering). See `examples/tabular_preprocessing.py` or configure explicitly:

```python
from condensite_torch import CondensiteTorchCDEConfig, TabularPreprocessorConfig

config = CondensiteTorchCDEConfig(
    preprocessor=TabularPreprocessorConfig(
        add_missing_indicator=True,
        handle_unknown="use_unknown",
    ),
)
```
- **Sampling benchmark**: `scripts/aux_sampling_benchmark.py` trains a small model with `sampler in {iid, stratified, lhs, sobol, importance}` and prints JSON means/std-devs of CRPS/NLL so you can quantify the trade-offs.
- **Benchmark suite**: `python -m benchmarks.run` trains Condensite alongside Gaussian + quantile baselines on heteroscedastic/multimodal datasets and writes JSON metrics (default `benchmarks/results.json`). Pass `--quick` for a CI-friendly downsampled run or `--output` to control the artifact path; the quick mode is what CI executes.
- **Early stopping**: Use `val_fraction>0` or pass `(X_val, y_val)` along with `patience` / `monitor_metric` to enable validation-driven checkpoints; `examples/early_stopping.py` shows how to inspect the recorded metrics and restored epoch.
- **Calibration diagnostics**: `scripts/calibration_report.py` emits PIT histograms and coverage stats so you can monitor probabilistic calibration over time.
- **Split-conformal intervals**: Wrap the estimator with `ConformalCDEWrapper` to obtain finite-sample predictive intervals, choosing `method="quantile"` or `"cdf"` for calibration style; see `examples/conformal_intervals.py`.
- **AMP & GPU**: Set `amp=True` when training on CUDA devices; automatic casting and gradient scaling are enabled through PyTorch AMP.

## Automated Tuning

```python
from condensite_cde.tune import tune_bandwidth_m_aux
from condensite_torch import CondensiteTorchCDEConfig
from condensite_torch.validation import SchemaConstraints

result = tune_bandwidth_m_aux(
    X_train,
    y_train,
    bandwidths=[0.08, 0.12, 0.16],
    m_aux_values=[64, 96],
    base_config=CondensiteTorchCDEConfig(
        epochs=6,
        patience=2,
        val_fraction=0.2,
        input_schema=SchemaConstraints(y_min=-1.0, y_max=1.0),
    ),
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
- `estimator.model_card()` reveals the training metadata/config hash/version info captured during fit (and persisted inside `metadata.json`).

## Examples

Run the scripts with Poetry to explore practical settings:

```bash
poetry run python examples/basic_tabular.py
poetry run python examples/tuning_bandwidth.py
poetry run python examples/compare_aux_sampling.py
poetry run python examples/tune_bandwidth.py
poetry run python examples/decision_metrics.py
poetry run python examples/conformal_intervals.py
poetry run python examples/quantiles_and_intervals.py
poetry run python examples/tail_risk.py
poetry run python examples/permutation_importance.py
poetry run python examples/what_if.py
poetry run python examples/tabular_preprocessing.py
poetry run python examples/multi_target.py
poetry run python examples/distribution_comparison.py
poetry run python examples/local_grids.py
poetry run python examples/epistemic_ensemble.py
poetry run python examples/custom_kernel_loss.py
poetry run python scripts/local_grid_benchmark.py
poetry run python scripts/inference_benchmark.py --row-batch 64 --grid-chunk 64
poetry run python -m benchmarks.run --quick
condensite fit --train data/train.csv --target target --output-model artifacts/model
condensite predict --model artifacts/model --data data/inference.csv --target target --output preds.csv
condensite report --model artifacts/model --data data/val.csv --target target --output-json reports/metrics.json
```

## Release process

See `docs/RELEASE.md` for the full checklist (version bump, tagging, and the trusted-publishing workflow that pushes artifacts to PyPI on `v*` tags).
```bash
poetry run python scripts/aux_sampling_benchmark.py > benchmark.json
poetry run python scripts/calibration_report.py
poetry run python -m benchmarks.run --datasets heteroscedastic,multimodal --output reports/benchmarks.json
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
- **Multi-target outputs**: `MultiTargetCondensite` wraps one or more estimators to model each target dimension independently or autoregressively; `examples/multi_target.py` demonstrates training and sampling with correlated targets.
- **Distribution comparison**: `condensite_torch.distribution_metrics` exposes Wasserstein-1, Kolmogorov–Smirnov, and Jensen–Shannon distances to compare two predicted distributions on the same grid; run `examples/distribution_comparison.py` to see them in action.
- **Local grids**: `condensite_torch.make_local_grid` builds per-row grids using estimated quantiles so inference focuses on the relevant range; see `examples/local_grids.py`. Defaults (`q_low=0.01`, `q_high=0.99`, `padding=0.1`) work well as a starting point—tighten the quantiles for faster inference on well-behaved targets or loosen them plus more padding for heavy tails.
- **Ensembles**: `EnsembleCondensite` trains multiple seeds/bootstraps and returns mean/variance for densities/quantiles (see `examples/epistemic_ensemble.py`) to capture epistemic uncertainty.
- **CLI**: `condensite` exposes `fit`, `predict`, and `report` subcommands so you can train/evaluate from CSV/Parquet without writing Python; see the quickstart commands below.
- **Model export**: `condensite_torch.export_torchscript` and `export_onnx` (optional dependency) trace any `nn.Module` with a sample tensor and save it for deployment; explicit preprocessing/feature concatenation must be handled by the caller.
- **Monitoring**: `condensite_torch.monitoring` exposes PSI/KS and PIT drift helpers with configurable warn/alert thresholds; `scripts/monitor_report.py` writes a schema-stable JSON payload with per-feature statuses for dashboards.

### Local grid best practices

- Run `poetry run python scripts/local_grid_benchmark.py --datasets heteroscedastic,multimodal --grid-size 64` to capture runtime/accuracy deltas between global and local grids on representative datasets; the script writes `reports/local_grid_benchmark.json` with speedups and NLL/CRPS deltas.
- Start with `q_low=0.01`, `q_high=0.99`, and `padding=0.1` for noisy/heteroscedastic targets. Tighten to `(0.05, 0.95)` without padding for well-behaved unimodal data, or widen to `(0.001, 0.999)` plus padding `>=0.2` when heavy tails or outliers are expected.
- When calling inference APIs (`predict_density`, `predict_cdf`, `evaluate`), set `use_local_grid=True` and optionally override `grid_size/q_low/q_high/padding` with `local_grid_params`. The estimator now caches the generated grids per dataset window so repeated evaluations avoid recomputing quantiles.
