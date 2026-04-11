from __future__ import annotations

from pathlib import Path

import numpy as np

from rasptorch.data_validation import apply_preprocessing, validate_dataset_structure


def test_validate_dataset_structure_reports_missing_columns(tmp_path: Path) -> None:
    rows = [
        {"feature": "1.0", "label": "cat", "path": str(tmp_path / "a.png")},
        {"feature": "2.0", "label": "dog", "path": str(tmp_path / "b.png")},
    ]

    out = validate_dataset_structure(
        rows,
        required_columns=["feature", "missing_col"],
        label_column="label",
        image_path_column="path",
        check_paths=True,
    )

    assert out["ok"] is False
    assert any("missing required column" in e.lower() for e in out["errors"])
    assert any("image path" in w.lower() for w in out["warnings"])


def test_validate_dataset_structure_reports_class_imbalance() -> None:
    rows = [{"x": i, "label": "A"} for i in range(20)] + [{"x": 999, "label": "B"}]
    out = validate_dataset_structure(rows, required_columns=["x", "label"], label_column="label")
    assert out["ok"] is True
    assert any("imbalance" in w.lower() for w in out["warnings"])


def test_apply_preprocessing_resize_and_normalize() -> None:
    x = np.random.RandomState(0).randn(2, 16, 16, 3).astype(np.float32)
    y, stats = apply_preprocessing(x, normalize="dataset", resize_hw=(8, 8))
    assert y.shape == (2, 8, 8, 3)
    assert stats["normalize"] == "dataset"
    assert stats["resize_hw"] == (8, 8)
    assert "mean" in stats and "std" in stats
