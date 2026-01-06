# Condensite Torch Documentation

## Quickstart

Train a model with automatic preprocessing and predict quantiles/tail probabilities:

```python
import numpy as np
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig, TabularPreprocessorConfig

X = np.array(
    [
        [1.2, "red", None],
        [0.5, "blue", "cat"],
        [np.nan, None, "dog"],
    ],
    dtype=object,
)
y = 0.3 * np.random.randn(X.shape[0])

config = CondensiteTorchCDEConfig(
    m_aux=16,
    epochs=4,
    patience=2,
    sampler="sobol",
    preprocessor=TabularPreprocessorConfig(add_missing_indicator=True),
)
estimator = CondensiteTorchCDE(config=config, random_seed=7).fit(X, y)

grid = np.linspace(y.min() - 1.0, y.max() + 1.0, 64)
pdf = estimator.predict_density(X, grid)
quantiles = estimator.predict_quantile(X, [0.1, 0.5, 0.9], y_grid=grid)
tail = estimator.predict_tail_prob(X, threshold=0.0, y_grid=grid)
print(pdf.shape, quantiles.shape, tail.shape)
```

## Probabilistic outputs

```python
cdf = estimator.predict_cdf(X, grid)
interval_lo, interval_hi = estimator.predict_interval(X, coverage=0.9, y_grid=grid)
```

## Reproducible training

```python
config = CondensiteTorchCDEConfig(reproducible=True, epochs=4, patience=2)
estimator = CondensiteTorchCDE(config=config, random_seed=7).fit(X, y)
card = estimator.model_card()
print(card["config_hash"], card["training"]["time_seconds"])
```

This mode sets deterministic seeds (Python/NumPy/Torch), pins torch deterministic algorithms, and stores a model card with schema stats, runtime versions, and early-stopping metadata. The same card is embedded in `metadata.json` when saving/restoring the estimator.

## Input validation

```python
from condensite_torch.validation import SchemaConstraints

schema = SchemaConstraints(
    numeric_indices=[0],
    categorical_indices=[1],
    allow_missing_numeric=False,
    max_categorical_cardinality=64,
    y_min=-1.0,
    y_max=1.0,
)
config = CondensiteTorchCDEConfig(input_schema=schema)
estimator = CondensiteTorchCDE(config=config).fit(X, y)  # raises ValidationError if schema violated
```

All inference APIs call `validate_inputs` internally, so a bad batch or unexpected data drift produces an actionable `ValidationError` instead of silently corrupting downstream metrics.

## Custom kernels & losses

```python
custom_config = CondensiteTorchCDEConfig(
    kernel="epanechnikov",
    loss="mae",
    m_aux=24,
    epochs=5,
)
custom_estimator = CondensiteTorchCDE(config=custom_config, random_seed=9).fit(X, y)
```

The `kernel` and `loss` knobs let you swap in a compact-support Epanechnikov smoother or MAE objective without modifying the trainer; see `examples/custom_kernel_loss.py` for a complete script.

## Fast inference

```python
fast_config = CondensiteTorchCDEConfig(
    inference_batch_size=64,
    inference_grid_chunk_size=64,
)
estimator = CondensiteTorchCDE(config=fast_config, random_seed=2).fit(X, y)
pdf = estimator.predict_density(X_test, grid)  # streams rows/grids in chunks under the hood
```

Tune `inference_batch_size` (rows) and `inference_grid_chunk_size` (grid points) when predicting on large datasets or dense grids to keep memory flat while maintaining accuracy. `scripts/inference_benchmark.py` prints before/after timings.

## Calibration + Conformal

```python
from condensite_torch.conformal import ConformalCDEWrapper

wrapper = ConformalCDEWrapper(config, random_seed=11).fit(X_train, y_train, X_cal, y_cal)
lower, upper = wrapper.predict_interval(X_test, coverage=0.9)
```

## CLI Walkthrough

```bash
condensite fit --train data/train.csv --target target --output-model artifacts/model
condensite predict --model artifacts/model --data data/inference.csv --target target --output preds.csv
condensite report --model artifacts/model --data data/val.csv --target target --output-json reports/metrics.json
```

## Benchmarks

```bash
poetry run python -m benchmarks.run --quick
poetry run python -m benchmarks.run --datasets heteroscedastic,multimodal,heavy_tail --output reports/benchmarks.json
```

The first command runs the CI-friendly quick mode; the second expands the dataset list and writes the JSON summary wherever you like (default is `benchmarks/results.json`, which is already ignored by git).
