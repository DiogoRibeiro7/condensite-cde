"""Apply the final review and strict-lint fixes on PR #14."""

from __future__ import annotations

from pathlib import Path


def _replace(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def _estimator_fixes() -> None:
    path = "src/condensite_torch/estimator.py"
    _replace(
        path,
        "        self.config = config or CondensiteTorchCDEConfig()\n"
        "        self.random_seed = int(random_seed)\n",
        "        self.config = config or CondensiteTorchCDEConfig()\n"
        "        if self.config.epistemic_mode == \"ensemble\":\n"
        "            msg = (\n"
        "                \"epistemic_mode='ensemble' is not implemented by CondensiteTorchCDE; \"\n"
        "                \"use EnsembleCondensite for ensemble uncertainty.\"\n"
        "            )\n"
        "            raise ValueError(msg)\n"
        "        self.random_seed = int(random_seed)\n",
    )
    _replace(
        path,
        "        *,\n"
        "        head: int | str | None = None,\n"
        "    ) -> NDArray[np.float64]:\n"
        "        assert self.x_scaler is not None\n",
        "        *,\n"
        "        head: int | str | None = None,\n"
        "        force_eval: bool = True,\n"
        "    ) -> NDArray[np.float64]:\n"
        "        assert self.x_scaler is not None\n",
    )
    _replace(
        path,
        "        assert self.model is not None\n"
        "        y_scaler = self.y_scaler\n"
        "        model = self.model\n"
        "        X_arr = np.asarray(X, dtype=object)\n",
        "        assert self.model is not None\n"
        "        X_arr = np.asarray(X, dtype=object)\n",
    )
    _replace(
        path,
        "            return self._predict_density_local(x_tensor, grid_arr, head=head)\n",
        "            return self._predict_density_local(\n"
        "                x_tensor, grid_arr, head=head, force_eval=force_eval\n"
        "            )\n",
    )
    _replace(
        path,
        "        outputs: list[Tensor] = []\n"
        "        self.model.eval()\n"
        "        with torch.no_grad():\n",
        "        outputs: list[Tensor] = []\n"
        "        if force_eval:\n"
        "            self.model.eval()\n"
        "        with torch.no_grad():\n",
    )
    _replace(
        path,
        "                pdf = self._predict_density_internal(X_arr, y_grid_arr, head=head)\n",
        "                pdf = self._predict_density_internal(\n"
        "                    X_arr, y_grid_arr, head=head, force_eval=False\n"
        "                )\n",
    )
    _replace(
        path,
        "        *,\n"
        "        head: int | str | None,\n"
        "    ) -> NDArray[np.float64]:\n"
        "        assert self.y_scaler is not None\n",
        "        *,\n"
        "        head: int | str | None,\n"
        "        force_eval: bool = True,\n"
        "    ) -> NDArray[np.float64]:\n"
        "        assert self.y_scaler is not None\n",
    )
    _replace(
        path,
        "        scale = max(float(y_scaler.data_range_), 1e-8)\n"
        "        model.eval()\n"
        "        with torch.no_grad():\n",
        "        scale = max(float(y_scaler.data_range_), 1e-8)\n"
        "        if force_eval:\n"
        "            model.eval()\n"
        "        with torch.no_grad():\n",
    )
    _replace(
        path,
        "        if size is None or size <= 0:\n"
        "            return total_rows\n"
        "        return max(1, min(total_rows, int(size)))\n",
        "        if size is None or size <= 0:\n"
        "            return max(1, total_rows)\n"
        "        return max(1, min(max(1, total_rows), int(size)))\n",
    )
    _replace(
        path,
        "    def _combine_features(self, X_batch: Tensor, y_prime_chunk: Tensor) -> Tensor:\n",
        "    @staticmethod\n"
        "    def _combine_features(X_batch: Tensor, y_prime_chunk: Tensor) -> Tensor:\n",
    )
    _replace(
        path,
        '        """Build schema constraints using fitted preprocessors and target range."""\n',
        '        """Build schema constraints using fitted preprocessing metadata."""\n',
    )
    _replace(
        path,
        "        y_min = float(np.min(y)) if y.size else None\n"
        "        y_max = float(np.max(y)) if y.size else None\n"
        "        return SchemaConstraints(\n",
        "        return SchemaConstraints(\n",
    )
    _replace(
        path,
        "            allow_missing_numeric=True,\n"
        "            allow_missing_categorical=True,\n"
        "            y_min=y_min,\n"
        "            y_max=y_max,\n"
        "        )\n",
        "            allow_missing_numeric=True,\n"
        "            allow_missing_categorical=True,\n"
        "            y_min=None,\n"
        "            y_max=None,\n"
        "        )\n",
    )


def _ensemble_fix() -> None:
    _replace(
        "src/condensite_torch/ensemble.py",
        "            config = copy.deepcopy(self.base_config)\n"
        "            estimator = CondensiteTorchCDE(config=config, random_seed=self.random_seed + idx)\n",
        "            config = copy.deepcopy(self.base_config)\n"
        "            if config.epistemic_mode == \"ensemble\":\n"
        "                config.epistemic_mode = \"none\"\n"
        "            estimator = CondensiteTorchCDE(config=config, random_seed=self.random_seed + idx)\n",
    )


def _lint_fixes() -> None:
    _replace(
        "scripts/monitor_report.py",
        "def main() -> None:\n",
        "def main() -> None:  # noqa: PLR0914\n",
    )

    path = "src/condensite_cde/adapters.py"
    _replace(
        path,
        "    def _extra_state(self) -> dict[str, Any]:  # pragma: no cover - overridden by subclasses\n",
        "    def _extra_state(self) -> dict[str, Any]:  # noqa: PLR6301  # pragma: no cover\n",
    )
    _replace(
        path,
        "    def _load_extra_state(self, payload: Mapping[str, Any]) -> None:  # pragma: no cover - optional\n",
        "    def _load_extra_state(self, payload: Mapping[str, Any]) -> None:  # noqa: PLR6301  # pragma: no cover\n",
    )
    _replace(
        path,
        "class SklearnCondensiteRegressor(_AdapterBase):\n"
        "    \"\"\"Sklearn-style wrapper around CondensiteTorchCDE.\"\"\"\n\n"
        "    def __init__(\n",
        "class SklearnCondensiteRegressor(_AdapterBase):\n"
        "    \"\"\"Sklearn-style wrapper around CondensiteTorchCDE.\"\"\"\n\n"
        "    def __init__(  # noqa: PLR0913\n",
    )
    _replace(
        path,
        "    def set_params(self, **params: Any) -> SklearnCondensiteRegressor:\n",
        "    def set_params(self, **params: Any) -> SklearnCondensiteRegressor:  # noqa: PLR0912\n",
    )
    _replace(
        path,
        "    def _require_pandas(self) -> None:\n",
        "    @staticmethod\n"
        "    def _require_pandas() -> None:\n",
    )

    path = "src/condensite_cde/cv.py"
    _replace(
        path,
        '_DEFAULT_METRICS: tuple[str, ...] = ("nll", "crps", "coverage")\n',
        '_DEFAULT_METRICS: tuple[str, ...] = ("nll", "crps", "coverage")\n'
        "_MIN_CV_FOLDS = 2\n",
    )
    _replace(
        path,
        "def cross_validate(  # noqa: PLR0913\n",
        "def cross_validate(  # noqa: PLR0913, PLR0914, PLR0915\n",
    )
    _replace(path, "    if cv < 2:\n", "    if cv < _MIN_CV_FOLDS:\n")
    _replace(
        path,
        "def _compute_metrics(\n",
        "def _compute_metrics(  # noqa: PLR0913, PLR0917\n",
    )
    _replace(
        path,
        "    bin_count = min(max(cv * 2, 2), min(unique.size, 20))\n",
        "    bin_count = min(max(cv * _MIN_CV_FOLDS, _MIN_CV_FOLDS), unique.size, 20)\n",
    )

    path = "src/condensite_cde/reports.py"
    _replace(
        path,
        "from datetime import datetime, timezone\n",
        "from datetime import datetime, timezone\nfrom itertools import pairwise\n",
    )
    _replace(
        path,
        "    if any(right <= left for left, right in zip(bin_edges, bin_edges[1:], strict=True)):\n",
        "    if any(right <= left for left, right in pairwise(bin_edges)):\n",
    )

    path = "src/condensite_cde/tune.py"
    _replace(
        path,
        "def tune_bandwidth_m_aux(  # noqa: PLR0913\n",
        "def tune_bandwidth_m_aux(  # noqa: PLR0913, PLR0914\n",
    )
    _replace(
        path,
        "def _build_run_metadata(\n",
        "def _build_run_metadata(  # noqa: PLR0913, PLR0917\n",
    )
    _replace(
        path,
        "def _clean_value(value: Any) -> Any:\n",
        "def _clean_value(value: Any) -> Any:  # noqa: PLR0911\n",
    )

    path = "src/condensite_torch/datasets.py"
    _replace(path, "def _load_delimited(\n", "def _load_delimited(  # noqa: PLR0914\n")
    _replace(
        path,
        '            if value is None or value.strip() == "":\n',
        "            if value is None or not value.strip():\n",
    )
    _replace(
        path,
        '    non_missing = [value for value in values if value.strip() != ""]\n',
        "    non_missing = [value for value in values if value.strip()]\n",
    )
    _replace(
        path,
        '        return [np.nan if value.strip() == "" else float(value) for value in values]\n',
        "        return [np.nan if not value.strip() else float(value) for value in values]\n",
    )
    _replace(
        path,
        '    return [None if value.strip() == "" else value for value in values]\n',
        "    return [None if not value.strip() else value for value in values]\n",
    )

    path = "src/condensite_torch/distribution_metrics.py"
    _replace(
        path,
        "_MIN_GRID_POINTS = 2\n",
        "_MIN_GRID_POINTS = 2\n_EXPECTED_ARRAY_DIMENSIONS = 2\n",
    )
    _replace(
        path,
        "    if arr_a.ndim != 2:\n",
        "    if arr_a.ndim != _EXPECTED_ARRAY_DIMENSIONS:\n",
    )

    path = "src/condensite_torch/metrics.py"
    _replace(
        path,
        "_MIN_GRID_POINTS = 2\n",
        "_MIN_GRID_POINTS = 2\n_ROW_LOCAL_GRID_DIMENSION = 2\n",
    )
    _replace(path, "def _validate_shapes(\n", "def _validate_shapes(  # noqa: PLR0912\n")
    _replace(
        path,
        "    elif y_grid_arr.ndim == 2:\n",
        "    elif y_grid_arr.ndim == _ROW_LOCAL_GRID_DIMENSION:\n",
    )

    path = "src/condensite_torch/monitoring.py"
    _replace(
        path,
        "_MIN_BINS = 2\n",
        "_MIN_BINS = 2\n_TWO_EDGE_CASE = 2\n",
    )
    _replace(path, "    if edges.size == 2:\n", "    if edges.size == _TWO_EDGE_CASE:\n")

    _replace(
        "tests/test_adapters.py",
        "    pdf_df = adapter.predict_density(frame.head(2))\n"
        "    assert pdf_df.shape[0] == 2\n",
        "    density_frame = frame.head(2)\n"
        "    pdf_df = adapter.predict_density(density_frame)\n"
        "    assert pdf_df.shape[0] == len(density_frame)\n",
    )
    _replace(
        "tests/test_cross_validation.py",
        "    assert len(result.folds) == 3\n",
        '    assert len(result.folds) == result.metadata["cv"]\n',
    )

    path = "tests/test_metrics.py"
    _replace(
        path,
        "from __future__ import annotations\n\nimport numpy as np\n",
        "from __future__ import annotations\n\nfrom itertools import pairwise\n\nimport numpy as np\n",
    )
    _replace(path, "np.trapezoid", "np.trapz")
    _replace(
        path,
        "    for earlier, later in zip(tail_probs, tail_probs[1:]):\n",
        "    for earlier, later in pairwise(tail_probs):\n",
    )

    path = "tests/unit/test_metrics.py"
    _replace(
        path,
        "    pdf = pdf / np.trapezoid(pdf, x=grid, axis=1)[:, None]\n",
        "    pdf /= np.trapz(pdf, x=grid, axis=1)[:, None]\n",
    )

    _replace(
        "tests/unit/test_monitoring.py",
        "    shifted = np.linspace(4.0, 6.0, 80)\n"
        "    assert population_stability_index(base, shifted) > 0.25\n",
        "    shifted = np.linspace(4.0, 6.0, 80)\n"
        "    minimum_detectable_psi = 0.25\n"
        "    assert population_stability_index(base, shifted) > minimum_detectable_psi\n",
    )
    _replace(
        "tests/unit/test_monitoring.py",
        "    stats = compare_windows(baseline, current, [\"a\", \"b\", \"c\"])\n"
        "    assert len(stats) == 3\n",
        "    feature_names = [\"a\", \"b\", \"c\"]\n"
        "    stats = compare_windows(baseline, current, feature_names)\n"
        "    assert len(stats) == len(feature_names)\n",
    )
    _replace(
        "tests/unit/test_reports.py",
        "    payload = build_benchmark_report(\n"
        '        results={"dataset": {"model": {"nll": 1.2, "crps": 0.5}}},\n',
        "    expected_nll = 1.2\n"
        "    payload = build_benchmark_report(\n"
        '        results={"dataset": {"model": {"nll": expected_nll, "crps": 0.5}}},\n',
    )
    _replace(
        "tests/unit/test_reports.py",
        '    assert payload["results"]["dataset"]["model"]["nll"] == 1.2\n',
        '    assert payload["results"]["dataset"]["model"]["nll"] == expected_nll\n',
    )


def _write_reviewer_tests() -> None:
    Path("tests/unit/test_reviewer_estimator_fixes.py").write_text(
        '''"""Regression tests for estimator findings raised during PR #12 review."""

from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
from torch import nn

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

pytestmark = pytest.mark.unit


def test_ensemble_epistemic_mode_fails_explicitly() -> None:
    config = CondensiteTorchCDEConfig(epistemic_mode="ensemble")
    with pytest.raises(ValueError, match="EnsembleCondensite"):
        CondensiteTorchCDE(config=config)


def test_empty_inference_batch_uses_positive_loop_step() -> None:
    estimator = CondensiteTorchCDE()
    minimum_batch_size = 1
    assert estimator._inference_row_batch_size(0) == minimum_batch_size


def test_derived_schema_does_not_freeze_training_target_range() -> None:
    estimator = CondensiteTorchCDE()
    X = np.zeros((3, 2), dtype=np.float64)
    y = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    schema = estimator._schema_from_training(X, y)
    assert schema is not None
    assert schema.y_min is None
    assert schema.y_max is None


def test_mc_dropout_preserves_training_mode_for_internal_passes() -> None:
    samples = 3
    rows = 2
    grid_points = 8
    config = CondensiteTorchCDEConfig(epistemic_mode="mc_dropout", mc_samples=samples)
    estimator = CondensiteTorchCDE(config=config)
    estimator._fitted = True
    estimator.model = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(1, 1))
    calls: list[tuple[bool, bool]] = []

    def fake_predict_density_internal(
        self: CondensiteTorchCDE,
        X: np.ndarray,
        y_grid: np.ndarray,
        *,
        head: int | str | None = None,
        force_eval: bool = True,
    ) -> np.ndarray:
        del head
        calls.append((force_eval, self.model.training if self.model is not None else False))
        return np.ones((X.shape[0], y_grid.size), dtype=np.float64)

    estimator._predict_density_internal = MethodType(fake_predict_density_internal, estimator)
    X = np.zeros((rows, 1), dtype=np.float64)
    grid = np.linspace(-1.0, 1.0, grid_points)
    output = estimator._predict_density_mc_dropout(X, grid)

    assert output.shape == (rows, grid_points)
    assert calls == [(False, True)] * samples
    assert estimator.model.training is True
''',
        encoding="utf-8",
    )


def main() -> None:
    _estimator_fixes()
    _ensemble_fix()
    _lint_fixes()
    _write_reviewer_tests()


if __name__ == "__main__":
    main()
