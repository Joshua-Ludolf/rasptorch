from __future__ import annotations

import numpy as np
import pytest

import rasptorch
from rasptorch.backends import CPUBackend


def test_backend_registry_contains_expected_backends() -> None:
    names = set(rasptorch.backend_manager.list_registered())
    assert {"cpu", "vulkan", "opencl", "cuda"}.issubset(names)


def test_connect_unknown_backend_raises() -> None:
    with pytest.raises(ValueError):
        rasptorch.connect_backend("not-a-backend")


def test_cpu_backend_math_matches_numpy() -> None:
    cpu = CPUBackend()
    cpu.initialize(strict=True)

    a = np.array([[1.0, -2.0], [3.5, 0.25]], dtype=np.float32)
    b = np.array([[2.0, 4.0], [-1.5, 2.0]], dtype=np.float32)

    np.testing.assert_allclose(cpu.add(a, b), a + b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cpu.mul(a, b), a * b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cpu.matmul(a, b.T), a @ b.T, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cpu.relu(a), np.maximum(a, 0.0), rtol=1e-6, atol=1e-6)


def test_connecting_optional_backends_is_safe_non_strict() -> None:
    try:
        chosen = rasptorch.connect_backend("vulkan", strict=False)
        assert chosen.name in {"vulkan", "cpu"}

        chosen = rasptorch.connect_backend("opencl", strict=False)
        assert chosen.name in {"opencl", "cpu"}

        chosen = rasptorch.connect_backend("cuda", strict=False)
        assert chosen.name in {"cuda", "cpu"}
    finally:
        rasptorch.connect_backend("cpu", strict=False)

