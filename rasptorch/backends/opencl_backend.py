from __future__ import annotations

import numpy as np

from ..backend import Backend


_OPENCL_KERNELS = """
__kernel void add(__global const float *a, __global const float *b, __global float *out, const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        out[gid] = a[gid] + b[gid];
    }
}

__kernel void mul(__global const float *a, __global const float *b, __global float *out, const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        out[gid] = a[gid] * b[gid];
    }
}

__kernel void relu(__global const float *x, __global float *out, const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float value = x[gid];
        out[gid] = value > 0.0f ? value : 0.0f;
    }
}

__kernel void matmul(
    __global const float *a,
    __global const float *b,
    __global float *out,
    const int rows,
    const int cols,
    const int inner
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < rows && col < cols) {
        float acc = 0.0f;
        for (int k = 0; k < inner; ++k) {
            acc += a[row * inner + k] * b[k * cols + col];
        }
        out[row * cols + col] = acc;
    }
}
"""


class OpenCLBackend(Backend):
    """An OpenCL backend using PyOpenCL for GPU operations."""
    name = "opencl"

    def __init__(self, *, allow_fallback: bool = True) -> None:
        self.allow_fallback = bool(allow_fallback)
        self._available = False
        self._cl = None
        self._queue = None
        self._program = None
        self._kernel_add = None
        self._kernel_mul = None
        self._kernel_relu = None
        self._kernel_matmul = None

    def initialize(self, *, strict: bool = False) -> None:
        try:
            import pyopencl as cl  # type: ignore

            platforms = cl.get_platforms()
            devices = [d for p in platforms for d in p.get_devices()]
            if not devices:
                raise RuntimeError("No OpenCL devices detected")
            ctx = cl.Context(devices=[devices[0]])
            self._queue = cl.CommandQueue(ctx)
            self._cl = cl
            self._program = cl.Program(ctx, _OPENCL_KERNELS).build()
            self._kernel_add = cl.Kernel(self._program, "add")
            self._kernel_mul = cl.Kernel(self._program, "mul")
            self._kernel_relu = cl.Kernel(self._program, "relu")
            self._kernel_matmul = cl.Kernel(self._program, "matmul")
            self._available = True
        except Exception as e:
            self._available = False
            if strict:
                raise RuntimeError(f"OpenCL initialization failed: {e}") from e

    def shutdown(self) -> None:
        self._queue = None
        self._cl = None
        self._program = None
        self._kernel_add = None
        self._kernel_mul = None
        self._kernel_relu = None
        self._kernel_matmul = None
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

    def _array_to_buffer(self, array: np.ndarray):
        if self._cl is None or self._queue is None:
            raise RuntimeError("OpenCL backend unavailable")
        data = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        return self._cl.Buffer(
            self._queue.context,
            self._cl.mem_flags.READ_ONLY | self._cl.mem_flags.COPY_HOST_PTR,
            hostbuf=data,
        ), data

    def _empty_buffer(self, nbytes: int):
        if self._cl is None or self._queue is None:
            raise RuntimeError("OpenCL backend unavailable")
        return self._cl.Buffer(self._queue.context, self._cl.mem_flags.WRITE_ONLY, size=nbytes)

    def _download(self, buffer, shape: tuple[int, ...]) -> np.ndarray:
        if self._cl is None or self._queue is None:
            raise RuntimeError("OpenCL backend unavailable")
        out = np.empty(shape, dtype=np.float32)
        self._cl.enqueue_copy(self._queue, out, buffer)
        return out

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            self._fallback_or_raise("OpenCL backend unavailable")
            return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)
        lhs, rhs = np.broadcast_arrays(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))
        flat_lhs = np.ascontiguousarray(lhs.reshape(-1))
        flat_rhs = np.ascontiguousarray(rhs.reshape(-1))
        buf_a, _ = self._array_to_buffer(flat_lhs)
        buf_b, _ = self._array_to_buffer(flat_rhs)
        buf_out = self._empty_buffer(flat_lhs.nbytes)
        self._kernel_add(
            self._queue,
            (int(flat_lhs.size),),
            None,
            buf_a,
            buf_b,
            buf_out,
            np.int32(flat_lhs.size),
        )
        return np.asarray(self._download(buf_out, lhs.shape), dtype=np.float32)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            self._fallback_or_raise("OpenCL backend unavailable")
            return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)
        lhs, rhs = np.broadcast_arrays(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))
        flat_lhs = np.ascontiguousarray(lhs.reshape(-1))
        flat_rhs = np.ascontiguousarray(rhs.reshape(-1))
        buf_a, _ = self._array_to_buffer(flat_lhs)
        buf_b, _ = self._array_to_buffer(flat_rhs)
        buf_out = self._empty_buffer(flat_lhs.nbytes)
        self._kernel_mul(
            self._queue,
            (int(flat_lhs.size),),
            None,
            buf_a,
            buf_b,
            buf_out,
            np.int32(flat_lhs.size),
        )
        return np.asarray(self._download(buf_out, lhs.shape), dtype=np.float32)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            self._fallback_or_raise("OpenCL backend unavailable")
            return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)
        lhs = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
        rhs = np.ascontiguousarray(np.asarray(b, dtype=np.float32))
        if lhs.ndim != 2 or rhs.ndim != 2:
            raise ValueError("OpenCL matmul expects two 2D arrays")
        if lhs.shape[1] != rhs.shape[0]:
            raise ValueError(f"matmul shape mismatch: {lhs.shape} @ {rhs.shape}")

        rows, inner = lhs.shape
        _, cols = rhs.shape
        buf_a, _ = self._array_to_buffer(lhs.reshape(-1))
        buf_b, _ = self._array_to_buffer(rhs.reshape(-1))
        buf_out = self._empty_buffer(rows * cols * np.dtype(np.float32).itemsize)
        self._kernel_matmul(
            self._queue,
            (int(rows), int(cols)),
            None,
            buf_a,
            buf_b,
            buf_out,
            np.int32(rows),
            np.int32(cols),
            np.int32(inner),
        )
        return np.asarray(self._download(buf_out, (rows, cols)), dtype=np.float32)

    def relu(self, x: np.ndarray) -> np.ndarray:
        if not self.is_available():
            self._fallback_or_raise("OpenCL backend unavailable")
            return np.maximum(np.asarray(x, dtype=np.float32), 0.0)
        values = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        flat_values = values.reshape(-1)
        buf_x, _ = self._array_to_buffer(flat_values)
        buf_out = self._empty_buffer(flat_values.nbytes)
        self._kernel_relu(self._queue, (int(flat_values.size),), None, buf_x, buf_out, np.int32(flat_values.size))
        return np.asarray(self._download(buf_out, values.shape), dtype=np.float32)

