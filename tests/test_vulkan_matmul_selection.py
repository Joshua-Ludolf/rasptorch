from __future__ import annotations

import numpy as np

from rasptorch import vulkan_backend as vk


class _Buf:
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = np.asarray(arr, dtype=np.float32)
        self.shape = self.arr.shape
        self.nbytes = int(self.arr.nbytes)
        self.buffer = 1
        self.memory = 1
        self.host = None


def test_matmul_fast_prefers_wide_tiled_for_large_shapes(monkeypatch) -> None:
    calls: list[str] = []

    def _ensure(_a, _b=None):
        return True

    def _empty(shape):
        return _Buf(np.zeros(shape, dtype=np.float32))

    def _wide_tiled(a, b, out):
        calls.append("wide_tiled")
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    def _tiled(a, b, out):
        calls.append("tiled")
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    def _vec4(a, b, out):
        calls.append("vec4")
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    def _basic(a, b, out):
        calls.append("basic")
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    monkeypatch.setattr(vk, "_ensure_vulkan_or_numpy", _ensure)
    monkeypatch.setattr(vk, "empty", _empty)
    monkeypatch.setattr(vk, "matmul_vec4_wide_tiled_into", _wide_tiled)
    monkeypatch.setattr(vk, "matmul_tiled_into", _tiled)
    monkeypatch.setattr(vk, "matmul_vec4_tiled_into", _tiled)
    monkeypatch.setattr(vk, "matmul_vec4_into", _vec4)
    monkeypatch.setattr(vk, "matmul_into", _basic)

    a = _Buf(np.ones((128, 128), dtype=np.float32))
    b = _Buf(np.ones((128, 128), dtype=np.float32))

    out = vk.matmul_fast(a, b)

    assert calls == ["wide_tiled"]
    assert out.shape == (128, 128)
    assert np.allclose(out.arr, a.arr @ b.arr)


def test_matmul_fast_uses_tiled_kernel_before_vec4_for_mid_size_shapes(monkeypatch) -> None:
    calls: list[str] = []

    def _ensure(_a, _b=None):
        return True

    def _empty(shape):
        return _Buf(np.zeros(shape, dtype=np.float32))

    def _tiled(a, b, out):
        calls.append("tiled")
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    def _vec4(a, b, out):
        calls.append("vec4")
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    def _basic(a, b, out):
        calls.append("basic")
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    monkeypatch.setattr(vk, "_ensure_vulkan_or_numpy", _ensure)
    monkeypatch.setattr(vk, "empty", _empty)
    monkeypatch.setattr(vk, "matmul_tiled_into", _tiled)
    monkeypatch.setattr(vk, "matmul_vec4_tiled_into", lambda _a, _b, _out: (_ for _ in ()).throw(RuntimeError("vec4 tiled should not be used first")))
    monkeypatch.setattr(vk, "matmul_vec4_into", _vec4)
    monkeypatch.setattr(vk, "matmul_into", _basic)

    a = _Buf(np.ones((64, 64), dtype=np.float32))
    b = _Buf(np.ones((64, 64), dtype=np.float32))

    out = vk.matmul_fast(a, b)

    assert calls == ["tiled"]
    assert out.shape == (64, 64)
    assert np.allclose(out.arr, a.arr @ b.arr)


def test_matmul_fast_uses_basic_kernel_for_small_non_vec4_shapes(monkeypatch) -> None:
    calls: list[str] = []

    def _ensure(_a, _b=None):
        return True

    def _empty(shape):
        return _Buf(np.zeros(shape, dtype=np.float32))

    def _tiled(a, b, out):
        calls.append("tiled")
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    def _vec4(a, b, out):
        calls.append("vec4")
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    def _basic(a, b, out):
        calls.append("basic")
        out.arr = np.asarray(a.arr @ b.arr, dtype=np.float32)
        return out

    monkeypatch.setattr(vk, "_ensure_vulkan_or_numpy", _ensure)
    monkeypatch.setattr(vk, "empty", _empty)
    monkeypatch.setattr(vk, "matmul_vec4_tiled_into", _tiled)
    monkeypatch.setattr(vk, "matmul_vec4_into", _vec4)
    monkeypatch.setattr(vk, "matmul_into", _basic)

    a = _Buf(np.ones((8, 6), dtype=np.float32))
    b = _Buf(np.ones((6, 5), dtype=np.float32))

    out = vk.matmul_fast(a, b)

    assert calls == ["basic"]
    assert out.shape == (8, 5)
    assert np.allclose(out.arr, a.arr @ b.arr)


def test_matmul_a_bt_fast_prefers_tiled_for_large_shapes(monkeypatch) -> None:
    calls: list[str] = []

    def _ensure(_a, _b=None):
        return True

    def _empty(shape):
        return _Buf(np.zeros(shape, dtype=np.float32))

    def _tiled(a, b, out):
        calls.append("tiled")
        out.arr = np.asarray(a.arr @ b.arr.T, dtype=np.float32)
        return out

    def _basic(a, b, out):
        calls.append("basic")
        out.arr = np.asarray(a.arr @ b.arr.T, dtype=np.float32)
        return out

    monkeypatch.setattr(vk, "_ensure_vulkan_or_numpy", _ensure)
    monkeypatch.setattr(vk, "empty", _empty)
    monkeypatch.setattr(vk, "matmul_a_bt_tiled_out", _tiled)
    monkeypatch.setattr(vk, "matmul_a_bt_out", _basic)

    a = _Buf(np.ones((128, 64), dtype=np.float32))
    b = _Buf(np.ones((96, 64), dtype=np.float32))

    out = vk.matmul_a_bt_fast(a, b)

    assert calls == ["tiled"]
    assert out.shape == (128, 96)
    assert np.allclose(out.arr, a.arr @ b.arr.T)


def test_matmul_at_b_fast_uses_transpose_helper(monkeypatch) -> None:
    calls: list[str] = []

    def _ensure(_a, _b=None):
        return True

    def _empty(shape):
        return _Buf(np.zeros(shape, dtype=np.float32))

    def _at_b(a, b, out):
        calls.append("at_b")
        out.arr = np.asarray(a.arr.T @ b.arr, dtype=np.float32)
        return out

    monkeypatch.setattr(vk, "_ensure_vulkan_or_numpy", _ensure)
    monkeypatch.setattr(vk, "empty", _empty)
    monkeypatch.setattr(vk, "matmul_at_b_out", _at_b)

    a = _Buf(np.ones((128, 64), dtype=np.float32))
    b = _Buf(np.ones((128, 32), dtype=np.float32))

    out = vk.matmul_at_b_fast(a, b)

    assert calls == ["at_b"]
    assert out.shape == (64, 32)
    assert np.allclose(out.arr, a.arr.T @ b.arr)
