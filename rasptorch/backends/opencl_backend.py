from __future__ import annotations

import numpy as np

from ..backend import Backend


class OpenCLBackend(Backend):
    """An OpenCL backend using PyOpenCL for GPU operations."""
    name = "opencl"

    def __init__(self, *, allow_fallback: bool = True) -> None:
        self.allow_fallback = bool(allow_fallback)
        self._available = False
        self._cl = None
        self._cl_array = None
        self._queue = None

    def initialize(self, *, strict: bool = False) -> None:
        try:
            import pyopencl as cl  # type: ignore
            import pyopencl.array as cl_array  # type: ignore

            platforms = cl.get_platforms()
            devices = [d for p in platforms for d in p.get_devices()]
            if not devices:
                raise RuntimeError("No OpenCL devices detected")
            ctx = cl.Context(devices=[devices[0]])
            self._queue = cl.CommandQueue(ctx)
            self._cl = cl
            self._cl_array = cl_array
            self._available = True
        except Exception as e:
            self._available = False
            if strict:
                raise RuntimeError(f"OpenCL initialization failed: {e}") from e

    def shutdown(self) -> None:
        self._queue = None
        self._cl = None
        self._cl_array = None
        self._available = False

    def is_available(self) -> bool:
        if self._available:
            return True
        self.initialize(strict=False)
        return self._available

    def _fallback_or_raise(self, msg: str) -> None:
        if self.allow_fallback:
            return
        raise RuntimeError(msg)

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            self._fallback_or_raise("OpenCL backend unavailable")
            return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)
        qa = self._cl_array.to_device(self._queue, np.asarray(a, dtype=np.float32))
        qb = self._cl_array.to_device(self._queue, np.asarray(b, dtype=np.float32))
        return np.asarray((qa + qb).get(), dtype=np.float32)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            self._fallback_or_raise("OpenCL backend unavailable")
            return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)
        qa = self._cl_array.to_device(self._queue, np.asarray(a, dtype=np.float32))
        qb = self._cl_array.to_device(self._queue, np.asarray(b, dtype=np.float32))
        return np.asarray((qa * qb).get(), dtype=np.float32)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            self._fallback_or_raise("OpenCL backend unavailable")
            return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)
        qa = self._cl_array.to_device(self._queue, np.asarray(a, dtype=np.float32))
        qb = self._cl_array.to_device(self._queue, np.asarray(b, dtype=np.float32))
        return np.asarray(self._cl_array.dot(qa, qb).get(), dtype=np.float32)

    def relu(self, x: np.ndarray) -> np.ndarray:
        if not self.is_available():
            self._fallback_or_raise("OpenCL backend unavailable")
            return np.maximum(np.asarray(x, dtype=np.float32), 0.0)
        qx = self._cl_array.to_device(self._queue, np.asarray(x, dtype=np.float32))
        return np.asarray(self._cl_array.maximum(qx, 0.0).get(), dtype=np.float32)

