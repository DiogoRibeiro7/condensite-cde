# condensite-cde

PyTorch tabular conditional density estimator utilities with a clean developer experience.

## Features

- Modern `src/`-layout with Poetry management (`>=3.10,<3.13`)
- Ruff and MyPy in strict mode keep quality high
- Ready-to-go CI pipeline plus GitHub templates and governance docs

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
- **Sampling strategy**: Stratified or Sobol typically outperform IID; `examples/compare_aux_sampling.py` demonstrates the trade-offs.
- **AMP & GPU**: Set `amp=True` when training on CUDA devices; automatic casting and gradient scaling are enabled through PyTorch AMP.

## Examples

Run the scripts with Poetry to explore practical settings:

```bash
poetry run python examples/basic_tabular.py
poetry run python examples/tuning_bandwidth.py
poetry run python examples/compare_aux_sampling.py
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
