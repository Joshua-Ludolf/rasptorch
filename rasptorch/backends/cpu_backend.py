from __future__ import annotations

import numpy as np

from ..backend import Backend


class CPUBackend(Backend):
    """A simple CPU backend using NumPy for operations."""
    name = "cpu"

    def initialize(self, *, strict: bool = False) -> None:
        return

    def shutdown(self) -> None:
        return

    def is_available(self) -> bool:
        return True

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(x, dtype=np.float32), 0.0)

