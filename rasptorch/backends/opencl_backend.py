from __future__ import annotations

import os
import re

import numpy as np

from ..backend import Backend


_OPENCL_KERNELS = """
// rasptorch OpenCL kernels

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

// Naive reference implementation (kept for correctness fallback)
__kernel void matmul_naive(
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

// Tiled matmul using local memory.
// Tile size is provided via build option: -DTS=<int>
#ifndef TS
#define TS 16
#endif

__kernel void matmul_tiled(
    __global const float *a,
    __global const float *b,
    __global float *out,
    const int rows,
    const int cols,
    const int inner
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int lr = get_local_id(0);
    const int lc = get_local_id(1);

    __local float As[TS][TS];
    __local float Bs[TS][TS];

    float acc = 0.0f;
    const int tiles = (inner + TS - 1) / TS;
    for (int t = 0; t < tiles; ++t) {
        const int kA = t * TS + lc;
        const int kB = t * TS + lr;

        As[lr][lc] = (row < rows && kA < inner) ? a[row * inner + kA] : 0.0f;
        Bs[lr][lc] = (kB < inner && col < cols) ? b[kB * cols + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TS; ++k) {
            acc += As[lr][k] * Bs[k][lc];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < rows && col < cols) {
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
        self._kernel_matmul_naive = None
        self._kernel_matmul_tiled = None
        self._matmul_tile: int | None = None
        self._device_info: str | None = None

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(str(raw).strip())
        except Exception:
            return default

    @staticmethod
    def _hint_match(text: str, hint: str | None) -> bool:
        if not hint:
            return True
        needle = str(hint).strip().lower()
        if not needle:
            return True
        return needle in str(text).lower()

    def _choose_device(self, cl):
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms detected")

        platform_hint = os.getenv("RASPTORCH_OPENCL_PLATFORM")
        device_hint = os.getenv("RASPTORCH_OPENCL_DEVICE")
        prefer = str(os.getenv("RASPTORCH_OPENCL_PREFER", "gpu")).strip().lower()

        candidates: list[tuple[object, object]] = []
        for p in platforms:
            try:
                devs = p.get_devices()
            except Exception:
                continue
            for d in devs:
                candidates.append((p, d))

        if not candidates:
            raise RuntimeError(
                "No OpenCL devices detected (on Raspberry Pi 5, install an OpenCL ICD like clvk to run OpenCL on the Vulkan GPU)"
            )

        filtered = [
            (p, d)
            for (p, d) in candidates
            if self._hint_match(getattr(p, "name", "") + " " + getattr(p, "vendor", ""), platform_hint)
            and self._hint_match(getattr(d, "name", "") + " " + getattr(d, "vendor", ""), device_hint)
        ]
        if filtered:
            candidates = filtered

        def _score(p, d) -> int:
            p_name = str(getattr(p, "name", "")).lower()
            p_vendor = str(getattr(p, "vendor", "")).lower()
            d_name = str(getattr(d, "name", "")).lower()
            d_vendor = str(getattr(d, "vendor", "")).lower()
            text = " ".join([p_name, p_vendor, d_name, d_vendor])

            score = 0
            if "clvk" in text or "vulkan" in text:
                # Helpful on systems like Raspberry Pi where OpenCL may be provided via clvk.
                # Kept as a modest bias so native vendor OpenCL can still win on desktops.
                score += 150
            if prefer in {"clvk", "vulkan"}:
                if "clvk" in text or "vulkan" in text:
                    score += 2000
            if prefer in {"gpu", "auto"}:
                # no-op: we always prefer GPU below
                pass

            try:
                if int(getattr(d, "type", 0)) & int(cl.device_type.GPU):
                    score += 800
                if int(getattr(d, "type", 0)) & int(cl.device_type.ACCELERATOR):
                    score += 500
                if int(getattr(d, "type", 0)) & int(cl.device_type.CPU):
                    score += 50
            except Exception:
                pass

            score += int(getattr(d, "max_compute_units", 0))
            score += int(getattr(d, "max_clock_frequency", 0) // 10)
            score += int(getattr(d, "global_mem_size", 0) // (256 * 1024 * 1024))
            if re.search(r"(llvmpipe|software)", text):
                score -= 10_000
            return score

        chosen_p, chosen_d = max(candidates, key=lambda pd: _score(pd[0], pd[1]))
        return chosen_p, chosen_d

    def _choose_tile(self, device) -> int:
        requested = max(1, int(self._env_int("RASPTORCH_OPENCL_TILE", 16)))
        try:
            max_wg = int(getattr(device, "max_work_group_size", 1))
        except Exception:
            max_wg = 1
        try:
            wis = getattr(device, "max_work_item_sizes", None)
            max_wi0 = int(wis[0]) if wis is not None and len(wis) >= 2 else 1
            max_wi1 = int(wis[1]) if wis is not None and len(wis) >= 2 else 1
        except Exception:
            max_wi0, max_wi1 = 1, 1

        for tile in (16, 8, 4, 2, 1):
            if tile > requested:
                continue
            if tile * tile <= max_wg and tile <= max_wi0 and tile <= max_wi1:
                return tile
        return 1

    def initialize(self, *, strict: bool = False) -> None:
        try:
            import pyopencl as cl  # type: ignore

            _platform, device = self._choose_device(cl)
            ctx = cl.Context(devices=[device])
            self._queue = cl.CommandQueue(
                ctx,
                properties=getattr(cl.command_queue_properties, "PROFILING_ENABLE", 0),
            )
            self._cl = cl

            tile = self._choose_tile(device)
            self._matmul_tile = tile
            build_opts = ["-cl-std=CL1.2", "-cl-fast-relaxed-math", f"-DTS={tile}"]
            self._program = cl.Program(ctx, _OPENCL_KERNELS).build(options=" ".join(build_opts))
            self._kernel_add = cl.Kernel(self._program, "add")
            self._kernel_mul = cl.Kernel(self._program, "mul")
            self._kernel_relu = cl.Kernel(self._program, "relu")
            self._kernel_matmul_naive = cl.Kernel(self._program, "matmul_naive")
            self._kernel_matmul_tiled = cl.Kernel(self._program, "matmul_tiled")
            self._device_info = f"{getattr(device, 'name', '')} ({getattr(device, 'vendor', '')})"
            self._available = True
        except Exception as e:
            self._available = False
            if strict:
                msg = str(e)
                # Common case: pyopencl is installed but no OpenCL ICD/platform is present.
                if "PLATFORM_NOT_FOUND_KHR" in msg or "No OpenCL platforms" in msg:
                    msg = (
                        f"{msg}\n"
                        "Hint: No OpenCL platform is visible. On Debian, install an OpenCL ICD (runtime), e.g.:\n"
                        "  sudo apt install -y pocl-opencl-icd   # CPU OpenCL (always works)\n"
                        "  sudo apt install -y mesa-opencl-icd   # Mesa OpenCL (may expose GPU if supported)\n"
                        "Then verify with: clinfo (Number of platforms should be > 0)."
                    )
                raise RuntimeError(f"OpenCL initialization failed: {msg}") from e

    def shutdown(self) -> None:
        self._queue = None
        self._cl = None
        self._program = None
        self._kernel_add = None
        self._kernel_mul = None
        self._kernel_relu = None
        self._kernel_matmul_naive = None
        self._kernel_matmul_tiled = None
        self._matmul_tile = None
        self._device_info = None
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

        tile = int(self._matmul_tile or 1)
        if tile > 1 and self._kernel_matmul_tiled is not None:
            g0 = int(((rows + tile - 1) // tile) * tile)
            g1 = int(((cols + tile - 1) // tile) * tile)
            self._kernel_matmul_tiled(
                self._queue,
                (g0, g1),
                (tile, tile),
                buf_a,
                buf_b,
                buf_out,
                np.int32(rows),
                np.int32(cols),
                np.int32(inner),
            )
        else:
            if self._kernel_matmul_naive is None:
                raise RuntimeError("OpenCL matmul kernel unavailable")
            self._kernel_matmul_naive(
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

