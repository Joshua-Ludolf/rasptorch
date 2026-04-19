from __future__ import annotations

from abc import ABC, abstractmethod
from threading import RLock
from typing import Dict

import numpy as np


class Backend(ABC):
    """Abstract compute backend contract used by backend adapters."""

    name: str

    @abstractmethod
    def initialize(self, *, strict: bool = False) -> None:
        """Initialize backend resources."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release backend resources."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when backend can execute compute operations."""

    @abstractmethod
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Elementwise addition."""

    @abstractmethod
    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Elementwise multiplication."""

    @abstractmethod
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication."""

    @abstractmethod
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""


class BackendManager:
    """Registry + active backend selection."""

    def __init__(self) -> None:
        self._registry: Dict[str, Backend] = {}
        self._active_name: str | None = None
        self._lock = RLock()

    def register(self, backend: Backend, *, set_default: bool = False) -> None:
        with self._lock:
            self._registry[backend.name] = backend
            if set_default or self._active_name is None:
                self._active_name = backend.name

    def get(self, name: str) -> Backend:
        with self._lock:
            try:
                return self._registry[name]
            except KeyError as e:
                raise ValueError(f"Unknown backend: {name}") from e

    def list_registered(self) -> list[str]:
        with self._lock:
            return sorted(self._registry.keys())

    def list_available(self) -> dict[str, bool]:
        with self._lock:
            return {name: backend.is_available() for name, backend in self._registry.items()}

    def connect(self, name: str, *, strict: bool = False) -> Backend:
        with self._lock:
            backend = self.get(name)
            backend.initialize(strict=strict)

            if backend.is_available():
                self._active_name = name
                return backend

            if strict:
                raise RuntimeError(f"Backend '{name}' is unavailable")

            if "cpu" not in self._registry:
                self._active_name = name
                return backend

            cpu = self._registry["cpu"]
            cpu.initialize(strict=False)
            self._active_name = "cpu"
            return cpu

    def active_backend(self) -> Backend:
        with self._lock:
            if self._active_name is None:
                raise RuntimeError("No backend is registered")
            return self._registry[self._active_name]

    def active_name(self) -> str:
        with self._lock:
            if self._active_name is None:
                raise RuntimeError("No backend is registered")
            return self._active_name

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.active_backend().add(a, b)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.active_backend().mul(a, b)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.active_backend().matmul(a, b)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return self.active_backend().relu(x)


def _build_default_manager() -> BackendManager:
    from .backends import CUDABackend, CPUBackend, OpenCLBackend, VulkanComputeBackend

    manager = BackendManager()
    manager.register(CPUBackend(), set_default=True)
    manager.register(VulkanComputeBackend())
    manager.register(OpenCLBackend())
    manager.register(CUDABackend())
    return manager


backend_manager = _build_default_manager()


def connect_backend(name: str, *, strict: bool = False) -> Backend:
    return backend_manager.connect(name, strict=strict)


def get_backend() -> Backend:
    return backend_manager.active_backend()


def available_backends() -> dict[str, bool]:
    return backend_manager.list_available()
