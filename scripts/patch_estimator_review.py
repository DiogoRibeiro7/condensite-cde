"""Apply the final estimator fixes requested by PR #12 review.

This file is temporary and is removed after the patch is committed.
"""

from __future__ import annotations

from pathlib import Path

ESTIMATOR_PATH = Path("src/condensite_torch/estimator.py")
TEST_PATH = Path("tests/unit/test_reviewer_estimator_fixes.py")


def _replace_once(text: str, old: str, new: str) -> str:
    """Replace one exact snippet, accepting an already-applied replacement."""
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        msg = f"Expected exactly one patch match, found {count}: {old[:80]!r}"
        raise RuntimeError(msg)
    return text.replace(old, new, 1)


def main() -> None:
    """Patch estimator semantics and write focused regression tests."""
    text = ESTIMATOR_PATH.read_text(encoding="utf-8")
    replacements = (
        (
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
        ),
        (
            "        *,\n"
            "        head: int | str | None = None,\n"
            "    ) -> NDArray[np.float64]:\n"
            "        assert self.x_scaler is not None\n",
            "        *,\n"
            "        head: int | str | None = None,\n"
            "        force_eval: bool = True,\n"
            "    ) -> NDArray[np.float64]:\n"
            "        assert self.x_scaler is not None\n",
        ),
        (
            "        outputs: list[Tensor] = []\n"
            "        self.model.eval()\n"
            "        with torch.no_grad():\n",
            "        outputs: list[Tensor] = []\n"
            "        if force_eval:\n"
            "            self.model.eval()\n"
            "        with torch.no_grad():\n",
        ),
        (
            "                pdf = self._predict_density_internal(X_arr, y_grid_arr, head=head)\n",
            "                pdf = self._predict_density_internal(\n"
            "                    X_arr,\n"
            "                    y_grid_arr,\n"
            "                    head=head,\n"
            "                    force_eval=False,\n"
            "                )\n",
        ),
        (
            "        if size is None or size <= 0:\n"
            "            return total_rows\n"
            "        return max(1, min(total_rows, int(size)))\n",
            "        if size is None or size <= 0:\n"
            "            return max(1, total_rows)\n"
            "        return max(1, min(max(1, total_rows), int(size)))\n",
        ),
        (
            "        y_min = float(np.min(y)) if y.size else None\n"
            "        y_max = float(np.max(y)) if y.size else None\n"
            "        return SchemaConstraints(\n",
            "        return SchemaConstraints(\n",
        ),
        (
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
        ),
    )
    for old, new in replacements:
        text = _replace_once(text, old, new)
    ESTIMATOR_PATH.write_text(text, encoding="utf-8")

    TEST_PATH.write_text(
        '''from __future__ import annotations

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
    assert estimator._inference_row_batch_size(0) == 1


def test_derived_schema_does_not_freeze_training_target_range() -> None:
    estimator = CondensiteTorchCDE()
    X = np.zeros((3, 2), dtype=np.float64)
    y = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    schema = estimator._schema_from_training(X, y)
    assert schema is not None
    assert schema.y_min is None
    assert schema.y_max is None


def test_mc_dropout_preserves_training_mode_for_internal_passes() -> None:
    config = CondensiteTorchCDEConfig(epistemic_mode="mc_dropout", mc_samples=3)
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
    X = np.zeros((2, 1), dtype=np.float64)
    grid = np.linspace(-1.0, 1.0, 8)
    output = estimator._predict_density_mc_dropout(X, grid)

    assert output.shape == (2, 8)
    assert calls == [(False, True), (False, True), (False, True)]
    assert estimator.model.training is True
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
