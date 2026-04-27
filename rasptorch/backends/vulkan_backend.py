from __future__ import annotations

import numpy as np

from ..backend import Backend
from .. import vulkan_backend as vk


class VulkanComputeBackend(Backend):
    """A Vulkan compute backend using the existing Vulkan module for GPU operations."""
    name = "vulkan"

    def __init__(self, *, allow_fallback: bool = True) -> None:
        self.allow_fallback = bool(allow_fallback)

    def initialize(self, *, strict: bool = False) -> None:
        vk.init(strict=strict)

    def shutdown(self) -> None:
        # Existing Vulkan module owns context lifecycle.
        return

    def is_available(self) -> bool:
        return vk.is_available()

    def _to_gpu(self, x: np.ndarray):
        return vk.to_gpu(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available() and self.allow_fallback:
            return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)

        abuf = self._to_gpu(a)
        bbuf = self._to_gpu(b)
        try:
            out = vk.add(abuf, bbuf)
            try:
                return np.asarray(vk.to_cpu(out), dtype=np.float32)
            finally:
                vk.free(out)
        finally:
            vk.free(abuf)
            vk.free(bbuf)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available() and self.allow_fallback:
            return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)

        abuf = self._to_gpu(a)
        bbuf = self._to_gpu(b)
        try:
            out = vk.mul(abuf, bbuf)
            try:
                return np.asarray(vk.to_cpu(out), dtype=np.float32)
            finally:
                vk.free(out)
        finally:
            vk.free(abuf)
            vk.free(bbuf)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available() and self.allow_fallback:
            return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)

        abuf = self._to_gpu(a)
        bbuf = self._to_gpu(b)
        try:
            out = vk.matmul_fast(abuf, bbuf)
            try:
                return np.asarray(vk.to_cpu(out), dtype=np.float32)
            finally:
                vk.free(out)
        finally:
            vk.free(abuf)
            vk.free(bbuf)

    def relu(self, x: np.ndarray) -> np.ndarray:
        if not self.is_available() and self.allow_fallback:
            return np.maximum(np.asarray(x, dtype=np.float32), 0.0)

        xbuf = self._to_gpu(x)
        try:
            out = vk.relu(xbuf)
            try:
                return np.asarray(vk.to_cpu(out), dtype=np.float32)
            finally:
                vk.free(out)
        finally:
            vk.free(xbuf)

