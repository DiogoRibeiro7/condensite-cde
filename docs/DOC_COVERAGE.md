# Documentation Coverage

## Inventory

- Enumerated source files with `rg --files` to ensure no modules were missed when assessing docstring coverage.

## Completed Modules

- `src/condensite_torch/estimator.py`: Public/critical helpers now include Google-style docstrings plus reproducibility notes.
- `src/condensite_torch/ensemble.py`: Constructor, prediction helpers, and persistence APIs fully documented.
- `src/condensite_torch/aux_sampling.py`: Internal samplers and `ImportanceSampler` document parameters/side effects.
- `src/condensite_cde/grids.py`: `make_y_grid` now clarifies arguments, clipping, and complexity.
- `src/condensite_torch/validation.py`: New schema + validation utilities describe every argument, error path, and complexity; inline comments highlight why certain guardrails exist (cardinality limits, missingness, etc.).
- `src/condensite_torch/losses.py`: Documents the pluggable loss registry so researchers can reason about element-wise reductions (`mse`, `mae`) without diving into the trainer.
- `benchmarks/datasets.py`, `benchmarks/run.py`, `benchmarks/run_all.py`, `benchmarks/models.py`: Benchmark utilities now specify expected inputs/outputs for every public function.

## Maintenance

- When new modules/functions are added, follow the same structure: imperative summary, Args/Returns/Raises/Side Effects/Complexity.
- Inline comments should explain *why* tricky logic exists (e.g., deterministic seeds, quantile interpolation guardrails).
- Re-run `rg --files` before declaring coverage complete for future large refactors.
