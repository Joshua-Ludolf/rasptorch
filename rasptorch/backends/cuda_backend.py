from __future__ import annotations

import numpy as np

from ..backend import Backend


class CUDABackend(Backend):
    """A CUDA backend using CuPy, with PyTorch CUDA fallback."""
    name = "cuda"

    def __init__(self, *, allow_fallback: bool = True) -> None:
        self.allow_fallback = bool(allow_fallback)
        self._cp = None
        self._torch = None
        self._provider: str | None = None
        self._available = False

    def initialize(self, *, strict: bool = False) -> None:
        self._cp = None
        self._torch = None
        self._provider = None
        self._available = False

        cupy_err: Exception | None = None
        try:
            import cupy as cp  # type: ignore

            count = int(cp.cuda.runtime.getDeviceCount())
            if count <= 0:
                raise RuntimeError("No CUDA devices detected")
            self._cp = cp
            self._provider = "cupy"
            self._available = True
            return
        except Exception as e:
            cupy_err = e

        try:
            import torch  # type: ignore

            if not bool(torch.cuda.is_available()):
                raise RuntimeError("torch.cuda.is_available() is False")
            self._torch = torch
            self._provider = "torch"
            self._available = True
            return
        except Exception as e:
            if strict:
                if cupy_err is not None:
                    raise RuntimeError(f"CUDA initialization failed (cupy={cupy_err}; torch={e})") from e
                raise RuntimeError(f"CUDA initialization failed: {e}") from e

    def shutdown(self) -> None:
        self._cp = None
        self._torch = None
        self._provider = None
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
            self._fallback_or_raise("CUDA backend unavailable")
            return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)
        if self._provider == "cupy":
            cp = self._cp
            if cp is None:
                self._fallback_or_raise("CuPy provider unavailable")
                return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)
            return np.asarray(
                cp.asnumpy(cp.asarray(a, dtype=cp.float32) + cp.asarray(b, dtype=cp.float32))
            )
        if self._provider == "torch":
            torch = self._torch
            if torch is None:
                self._fallback_or_raise("Torch CUDA provider unavailable")
                return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)
            ta = torch.as_tensor(a, device="cuda", dtype=torch.float32)
            tb = torch.as_tensor(b, device="cuda", dtype=torch.float32)
            return np.asarray((ta + tb).detach().cpu().numpy(), dtype=np.float32)
        self._fallback_or_raise("CUDA provider unavailable")
        return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            self._fallback_or_raise("CUDA backend unavailable")
            return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)
        if self._provider == "cupy":
            cp = self._cp
            if cp is None:
                self._fallback_or_raise("CuPy provider unavailable")
                return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)
            return np.asarray(
                cp.asnumpy(cp.asarray(a, dtype=cp.float32) * cp.asarray(b, dtype=cp.float32))
            )
        if self._provider == "torch":
            torch = self._torch
            if torch is None:
                self._fallback_or_raise("Torch CUDA provider unavailable")
                return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)
            ta = torch.as_tensor(a, device="cuda", dtype=torch.float32)
            tb = torch.as_tensor(b, device="cuda", dtype=torch.float32)
            return np.asarray((ta * tb).detach().cpu().numpy(), dtype=np.float32)
        self._fallback_or_raise("CUDA provider unavailable")
        return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            self._fallback_or_raise("CUDA backend unavailable")
            return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)
        if self._provider == "cupy":
            cp = self._cp
            if cp is None:
                self._fallback_or_raise("CuPy provider unavailable")
                return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)
            return np.asarray(
                cp.asnumpy(cp.asarray(a, dtype=cp.float32) @ cp.asarray(b, dtype=cp.float32))
            )
        if self._provider == "torch":
            torch = self._torch
            if torch is None:
                self._fallback_or_raise("Torch CUDA provider unavailable")
                return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)
            ta = torch.as_tensor(a, device="cuda", dtype=torch.float32)
            tb = torch.as_tensor(b, device="cuda", dtype=torch.float32)
            return np.asarray((ta @ tb).detach().cpu().numpy(), dtype=np.float32)
        self._fallback_or_raise("CUDA provider unavailable")
        return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)

    def relu(self, x: np.ndarray) -> np.ndarray:
        if not self.is_available():
            self._fallback_or_raise("CUDA backend unavailable")
            return np.maximum(np.asarray(x, dtype=np.float32), 0.0)
        if self._provider == "cupy":
            cp = self._cp
            if cp is None:
                self._fallback_or_raise("CuPy provider unavailable")
                return np.maximum(np.asarray(x, dtype=np.float32), 0.0)
            qx = cp.asarray(x, dtype=cp.float32)
            return np.asarray(cp.asnumpy(cp.maximum(qx, 0.0)))
        if self._provider == "torch":
            torch = self._torch
            if torch is None:
                self._fallback_or_raise("Torch CUDA provider unavailable")
                return np.maximum(np.asarray(x, dtype=np.float32), 0.0)
            tx = torch.as_tensor(x, device="cuda", dtype=torch.float32)
            return np.asarray(torch.relu(tx).detach().cpu().numpy(), dtype=np.float32)
        self._fallback_or_raise("CUDA provider unavailable")
        return np.maximum(np.asarray(x, dtype=np.float32), 0.0)

