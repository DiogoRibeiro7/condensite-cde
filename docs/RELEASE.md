## Release checklist

1. Update `pyproject.toml` / changelog with the new version (e.g., `0.1.1`). Commit the change.
2. Run the full test suite: `poetry run pytest -q -m "unit or integration or regression"`.
3. Build locally for sanity: `poetry build` (check that both wheel + sdist appear under `dist/`).
4. Tag the release and push:
   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```
5. GitHub Actions runs `.github/workflows/release.yml` which:
   - builds the artifacts with Poetry,
   - uploads them as workflow artifacts,
   - publishes to PyPI via trusted publishing (OIDC; no token needed).
6. Confirm on https://pypi.org/project/condensite-cde that the version is available and the CLI works:
   ```bash
   pip install condensite-cde==0.1.1
   python -c "import condensite_torch as ct; print(ct.__version__)"
   condensite --help
   ```
