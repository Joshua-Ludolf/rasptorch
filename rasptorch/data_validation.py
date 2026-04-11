from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ValidationIssue:
    level: str
    message: str


def _to_row_count(data: Any) -> int:
    if isinstance(data, Mapping):
        for v in data.values():
            try:
                return int(len(v))
            except Exception:
                continue
        return 0
    if isinstance(data, Sequence):
        return int(len(data))
    return 0


def validate_dataset_structure(
    data: Any,
    *,
    required_columns: Optional[Sequence[str]] = None,
    label_column: Optional[str] = None,
    image_path_column: Optional[str] = None,
    check_paths: bool = True,
) -> Dict[str, Any]:
    """Validate a tabular-like dataset payload.

    Supported input formats:
    - mapping of column -> sequence
    - sequence of row mappings
    """
    issues: list[ValidationIssue] = []
    warnings: list[str] = []
    errors: list[str] = []

    required = [str(c) for c in (required_columns or []) if str(c).strip()]

    rows = _to_row_count(data)
    if rows <= 0:
        errors.append("Dataset is empty")
        return {
            "ok": False,
            "rows": 0,
            "warnings": warnings,
            "errors": errors,
            "class_balance": {},
        }

    # Normalize access helpers for either supported shape.
    is_mapping = isinstance(data, Mapping)

    def has_col(name: str) -> bool:
        if is_mapping:
            return name in data
        first = data[0] if isinstance(data, Sequence) and data else {}
        return isinstance(first, Mapping) and name in first

    def col_values(name: str) -> Iterable[Any]:
        if is_mapping:
            vals = data.get(name, [])
            return vals if isinstance(vals, Sequence) else []
        out = []
        if isinstance(data, Sequence):
            for row in data:
                if isinstance(row, Mapping):
                    out.append(row.get(name))
        return out

    for col in required:
        if not has_col(col):
            errors.append(f"Missing required column: {col}")

    if label_column:
        if not has_col(label_column):
            errors.append(f"Missing label column: {label_column}")
        else:
            counts: Dict[str, int] = {}
            for v in col_values(label_column):
                k = str(v)
                counts[k] = counts.get(k, 0) + 1
            if counts:
                total = max(1, sum(counts.values()))
                ratios = {k: float(v / total) for k, v in counts.items()}
                max_ratio = max(ratios.values())
                min_ratio = min(ratios.values())
                # Warn when heavily imbalanced (largest class at least 4x smallest).
                if min_ratio > 0 and (max_ratio / min_ratio) >= 4.0:
                    warnings.append(
                        "Dataset imbalance detected: "
                        + ", ".join(f"{k}={ratios[k]*100:.1f}%" for k in sorted(ratios))
                    )
            else:
                ratios = {}
    else:
        ratios = {}

    if image_path_column:
        if not has_col(image_path_column):
            errors.append(f"Missing image path column: {image_path_column}")
        elif check_paths:
            missing = 0
            for v in col_values(image_path_column):
                p = Path(str(v))
                if not p.exists() or not p.is_file():
                    missing += 1
            if missing > 0:
                warnings.append(f"{missing} image path(s) do not exist or are not files")

    # Basic dtype warnings for numeric-like fields.
    numeric_cols = [c for c in required if c != image_path_column and c != label_column]
    for col in numeric_cols:
        vals = list(col_values(col))
        if not vals:
            continue
        sample = vals[0]
        if isinstance(sample, (str, bytes)):
            warnings.append(f"Column '{col}' appears non-numeric (sample type: {type(sample).__name__})")

    for msg in warnings:
        issues.append(ValidationIssue(level="warning", message=msg))
    for msg in errors:
        issues.append(ValidationIssue(level="error", message=msg))

    return {
        "ok": len(errors) == 0,
        "rows": rows,
        "warnings": warnings,
        "errors": errors,
        "issues": [issue.__dict__ for issue in issues],
        "class_balance": ratios,
    }


def _resize_nearest(images: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    n, in_h, in_w, c = images.shape
    y_idx = np.linspace(0, in_h - 1, out_h).astype(np.int64)
    x_idx = np.linspace(0, in_w - 1, out_w).astype(np.int64)
    out = images[:, y_idx][:, :, x_idx, :]
    assert out.shape == (n, out_h, out_w, c)
    return out


def apply_preprocessing(
    x: Any,
    *,
    normalize: str = "none",
    mean: Optional[float] = None,
    std: Optional[float] = None,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply a small, serializable preprocessing pipeline.

    - normalize='none': no normalization
    - normalize='dataset': use mean/std from data if not provided
    """
    arr = np.asarray(x, dtype=np.float32)
    stats: Dict[str, Any] = {"normalize": normalize, "resize_hw": resize_hw}

    if arr.ndim == 3:
        # Assume single image HWC.
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError("Expected input shaped [N,H,W,C] or [H,W,C]")

    if resize_hw is not None:
        h, w = int(resize_hw[0]), int(resize_hw[1])
        if h <= 0 or w <= 0:
            raise ValueError("resize_hw must contain positive dimensions")
        arr = _resize_nearest(arr, h, w)

    if str(normalize).lower() == "dataset":
        mu = float(np.mean(arr) if mean is None else mean)
        sigma = float(np.std(arr) if std is None else std)
        if sigma <= 0:
            sigma = 1.0
        arr = (arr - mu) / sigma
        stats["mean"] = mu
        stats["std"] = sigma

    return arr, stats
