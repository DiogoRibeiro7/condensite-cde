# Condensite Torch

PyTorch tabular conditional density estimator utilities with a clean developer experience.

## Features

- Modern `src/`-layout with Poetry management (`>=3.10,<3.13`)
- Ruff and MyPy in strict mode keep quality high
- Ready-to-go CI pipeline plus GitHub templates and governance docs

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
