from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterable, Optional, Set

import numpy as np

from . import vulkan_backend as vk


ArrayLike = Any


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer))


def _normalize_axes(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    normalized = []
    for ax in axes:
        ax_int = int(ax)
        if ax_int < 0:
            ax_int += ndim
        if ax_int < 0 or ax_int >= ndim:
            raise ValueError(f"axis {ax} is out of bounds for tensor with {ndim} dimensions")
        if ax_int in normalized:
            raise ValueError(f"duplicate axis {ax}")
        normalized.append(ax_int)
    return tuple(normalized)


def _normalize_dim(dim: int, ndim: int, *, allow_end: bool = False) -> int:
    dim_int = int(dim)
    upper = ndim if allow_end else (ndim - 1)
    lower = -ndim - (1 if allow_end else 0)
    if dim_int < lower or dim_int > upper:
        raise ValueError(f"dim {dim} is out of bounds for tensor with {ndim} dimensions")
    if dim_int < 0:
        dim_int += ndim + (1 if allow_end else 0)
    return dim_int


def _sum_to_shape(x: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Sum-reduce a broadcasted array back down to target_shape.

    Mirrors PyTorch/NumPy broadcasting gradient semantics.
    """

    target_shape = tuple(int(s) for s in target_shape)
    if x.shape == target_shape:
        return x

    out_shape = x.shape
    if len(target_shape) > len(out_shape):
        raise ValueError(f"cannot reduce grad shape {out_shape} to larger shape {target_shape}")

    padded_target = (1,) * (len(out_shape) - len(target_shape)) + target_shape
    reduce_axes: list[int] = []
    for i, (out_d, tgt_d) in enumerate(zip(out_shape, padded_target)):
        if tgt_d == 1 and out_d != 1:
            reduce_axes.append(i)
        elif tgt_d != out_d:
            raise ValueError(f"grad shape {out_shape} is not compatible with target {target_shape}")

    if reduce_axes:
        x = x.sum(axis=tuple(reduce_axes), keepdims=True)
    if len(out_shape) != len(target_shape):
        x = x.reshape(padded_target)
    return x.reshape(target_shape)


_grad_enabled: bool = True


def is_grad_enabled() -> bool:
    return _grad_enabled


@contextmanager
def set_grad_enabled(mode: bool):
    """Context manager to enable/disable autograd graph tracking."""

    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = bool(mode)
    try:
        yield
    finally:
        _grad_enabled = prev


@contextmanager
def no_grad():
    """Disable autograd graph tracking within the context (like torch.no_grad)."""

    with set_grad_enabled(False):
        yield


@contextmanager
def enable_grad():
    """Enable autograd graph tracking within the context (like torch.enable_grad)."""

    with set_grad_enabled(True):
        yield


class Tensor:
    """A minimal autograd-enabled tensor, backed by NumPy."""

    def __init__(
        self,
        data: ArrayLike,
        *,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] | None = None,
        _op: str = "",
        device: str = "cpu",
        _vkbuf: "vk.VulkanBuffer | None" = None,
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data

        self._vkbuf: "vk.VulkanBuffer | None" = _vkbuf
        # Keep a CPU-side ndarray for CPU tensors. For GPU tensors we keep a small
        # placeholder array to preserve shape/dtype expectations; real values are
        # fetched on demand via .numpy() / .to('cpu').
        if self._vkbuf is not None:
            self.data: np.ndarray = np.empty(self._vkbuf.shape, dtype=np.float32)
            device = "gpu"
        else:
            arr = np.array(data, dtype=np.float32)
            if device == "gpu":
                self._vkbuf = vk.to_gpu(arr)
                self.data = np.empty(self._vkbuf.shape, dtype=np.float32)
                device = "gpu"
            else:
                self.data = arr

        # CPU grads live in .grad (ndarray). GPU grads live in .grad_vkbuf.
        self.grad: Optional[np.ndarray] = None
        self.grad_vkbuf: "vk.VulkanBuffer | None" = None
        self.requires_grad: bool = requires_grad
        self.device: str = device
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set(_children) if _children is not None else set()
        self._op: str = _op

    @classmethod
    def _from_vkbuf(cls, buf: "vk.VulkanBuffer", *, requires_grad: bool = False) -> "Tensor":
        return cls(
            np.empty(buf.shape, dtype=np.float32),
            requires_grad=requires_grad,
            device="gpu",
            _vkbuf=buf,
        )

    def _accum_grad_vk(self, g: "vk.VulkanBuffer") -> None:
        if self.grad_vkbuf is None:
            self.grad_vkbuf = g
        else:
            prev = self.grad_vkbuf
            self.grad_vkbuf = vk.add(prev, g)
            vk.free(prev)

    def _as_vkbuf(self) -> "vk.VulkanBuffer":
        if self._vkbuf is not None:
            return self._vkbuf
        # Implicit upload: if a CPU tensor participates in a GPU op, promote it to a GPU tensor.
        new_vkbuf = vk.to_gpu(self.data)
        new_grad_vkbuf: "vk.VulkanBuffer | None" = None
        try:
            if self.grad is not None:
                new_grad_vkbuf = vk.to_gpu(self.grad)
            self._vkbuf = new_vkbuf
            self.grad_vkbuf = new_grad_vkbuf
            if new_grad_vkbuf is not None:
                self.grad = None
            self.data = np.empty(self._vkbuf.shape, dtype=np.float32)
            self.device = "gpu"
            return self._vkbuf
        except Exception:
            if new_grad_vkbuf is not None:
                vk.free(new_grad_vkbuf)
            vk.free(new_vkbuf)
            raise

    # ------------------------------------------------------------------
    # Basic tensor utilities
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"

    def detach(self) -> "Tensor":
        """Return a Tensor that shares storage but is detached from autograd."""

        if self._vkbuf is not None:
            return Tensor._from_vkbuf(self._vkbuf, requires_grad=False)
        return Tensor(self.data.copy(), requires_grad=False, device=self.device)

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        """In-place requires_grad flag set (leaf tensors only)."""

        if self._prev:
            raise RuntimeError("requires_grad_ can only be called on leaf tensors")
        self.requires_grad = bool(requires_grad)
        return self

    @property
    def shape(self) -> tuple[int, ...]:
        if self._vkbuf is not None:
            return self._vkbuf.shape
        return self.data.shape

    @property
    def T(self) -> "Tensor":
        """Transpose view (2D only for now)."""
        track = is_grad_enabled() and self.requires_grad
        if self.device == "gpu":
            vk_out = vk.transpose2d(self._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "transpose"
        else:
            out = Tensor(self.data.T, requires_grad=track, _children=(self,) if track else None, _op="transpose", device=self.device)

        if track:
            out._prev = {self}

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    self._accum_grad_vk(vk.transpose2d(out.grad_vkbuf))
                else:
                    if out.grad is None:
                        return
                    grad = out.grad.T
                    if self.grad is None:
                        self.grad = grad
                    else:
                        self.grad = self.grad + grad

            out._backward = _backward
        return out

    def view(self, *shape: int) -> "Tensor":
        """Reshape tensor without changing underlying storage (like torch.view)."""

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])  # type: ignore[assignment]
        new_shape = tuple(int(s) for s in shape)
        if int(np.prod(new_shape)) != int(np.prod(self.shape)):
            raise ValueError(f"view cannot change number of elements: {self.shape} -> {new_shape}")

        track = is_grad_enabled() and self.requires_grad

        if self.device == "gpu":
            vk_out = vk.view(self._as_vkbuf(), new_shape)
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "view"
        else:
            out = Tensor(self.data.reshape(new_shape), requires_grad=track, _children=(self,) if track else None, _op="view", device=self.device)

        if track:
            out._prev = {self}

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    self._accum_grad_vk(vk.view(out.grad_vkbuf, self.shape))
                else:
                    if out.grad is None:
                        return
                    g = out.grad.reshape(self.shape)
                    self.grad = g if self.grad is None else (self.grad + g)

            out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "Tensor":
        return self.view(*shape)

    def unsqueeze(self, dim: int) -> "Tensor":
        axis = _normalize_dim(dim, len(self.shape), allow_end=True)
        out_shape = self.shape[:axis] + (1,) + self.shape[axis:]
        return self.view(*out_shape)

    def squeeze(self, dim: int | None = None) -> "Tensor":
        if dim is None:
            out_shape = tuple(s for s in self.shape if s != 1)
            if not out_shape:
                out_shape = ()
            return self.view(*out_shape)

        axis = _normalize_dim(dim, len(self.shape))
        if self.shape[axis] != 1:
            return self
        out_shape = self.shape[:axis] + self.shape[axis + 1 :]
        return self.view(*out_shape)

    def permute(self, *dims: int) -> "Tensor":
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])  # type: ignore[assignment]
        if len(dims) != len(self.shape):
            raise ValueError(f"permute expected {len(self.shape)} dims, got {len(dims)}")
        axes = tuple(_normalize_dim(dim, len(self.shape)) for dim in dims)
        if len(set(axes)) != len(axes):
            raise ValueError("permute dims must be unique")

        track = is_grad_enabled() and self.requires_grad
        if self.device == "gpu":
            vk_out = vk.permute(self._as_vkbuf(), axes)
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "permute"
        else:
            out = Tensor(np.transpose(self.data, axes), requires_grad=track, _children=(self,) if track else None, _op="permute", device=self.device)

        if track:
            out._prev = {self}
            inv_axes = tuple(np.argsort(axes))

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    self._accum_grad_vk(vk.permute(out.grad_vkbuf, inv_axes))
                else:
                    if out.grad is None:
                        return
                    grad_in = np.transpose(out.grad, inv_axes)
                    self.grad = grad_in if self.grad is None else (self.grad + grad_in)

            out._backward = _backward
        return out

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        ndim = len(self.shape)
        a = _normalize_dim(dim0, ndim)
        b = _normalize_dim(dim1, ndim)
        if a == b:
            return self
        order = list(range(ndim))
        order[a], order[b] = order[b], order[a]
        return self.permute(order)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        ndim = len(self.shape)
        start = _normalize_dim(start_dim, ndim)
        end = _normalize_dim(end_dim, ndim)
        if start > end:
            raise ValueError(f"flatten expected start_dim <= end_dim, got {start_dim}, {end_dim}")
        merged = int(np.prod(self.shape[start : end + 1], dtype=np.int64))
        new_shape = self.shape[:start] + (merged,) + self.shape[end + 1 :]
        return self.view(*new_shape)

    # ------------------------------------------------------------------
    # Autograd core
    # ------------------------------------------------------------------
    def backward(self, grad: ArrayLike | None = None) -> None:
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar outputs")
            grad_arr = np.ones_like(self.data, dtype=self.data.dtype)
        else:
            grad_arr = np.array(grad, dtype=self.data.dtype)

        if self.device == "gpu":
            self.grad_vkbuf = vk.to_gpu(grad_arr)
        else:
            self.grad = grad_arr

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build_topo(t: Tensor) -> None:
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            t._backward()

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def _ensure_tensor(self, other: ArrayLike) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other, device=self.device)

    # Addition
    def __add__(self, other: ArrayLike) -> "Tensor":
        if _is_number(other):
            s = float(other)
            track = is_grad_enabled() and self.requires_grad
            if self.device == "gpu":
                vk_out = vk.add_scalar(self._as_vkbuf(), s)
                out = Tensor._from_vkbuf(vk_out, requires_grad=track)
                out._op = "+scalar"
            else:
                out = Tensor(self.data + s, requires_grad=track, _children=(self,) if track else None, _op="+scalar", device=self.device)

            if track:
                out._prev = {self}

                def _backward() -> None:
                    if out.device == "gpu":
                        if out.grad_vkbuf is None:
                            return
                        self._accum_grad_vk(out.grad_vkbuf)
                    else:
                        if out.grad is None:
                            return
                        self.grad = out.grad.copy() if self.grad is None else (self.grad + out.grad)

                out._backward = _backward
            return out

        other_t = self._ensure_tensor(other)
        # Limited broadcasting (PyTorch-like): (rows, cols) + (cols,)
        # Route through add_rowvec so gradients for the row-vector are correct.
        if len(self.shape) == 2 and len(other_t.shape) == 1 and other_t.shape[0] == self.shape[1]:
            return self.add_rowvec(other_t)

        track = is_grad_enabled() and (self.requires_grad or other_t.requires_grad)
        if self.device == "gpu" or other_t.device == "gpu":
            vk_out = vk.add(self._as_vkbuf(), other_t._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "+"
        else:
            out_data = self.data + other_t.data
            out = Tensor(out_data, requires_grad=track, _children=(self, other_t) if track else None, _op="+", device=self.device)

        if track:
            out._prev = {self, other_t}

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    # Handle broadcasting by reducing grad back to each operand's shape.
                    if self.shape == other_t.shape:
                        if self.requires_grad:
                            self._accum_grad_vk(out.grad_vkbuf)
                        if other_t.requires_grad:
                            other_t._accum_grad_vk(out.grad_vkbuf)
                        return

                    if self.requires_grad:
                        self._accum_grad_vk(vk.sum_to_shape(out.grad_vkbuf, self.shape))
                    if other_t.requires_grad:
                        other_t._accum_grad_vk(vk.sum_to_shape(out.grad_vkbuf, other_t.shape))
                else:
                    if out.grad is None:
                        return
                    if self.shape == other_t.shape:
                        if self.requires_grad:
                            if self.grad is None:
                                self.grad = out.grad.copy()
                            else:
                                self.grad = self.grad + out.grad
                        if other_t.requires_grad:
                            if other_t.grad is None:
                                other_t.grad = out.grad.copy()
                            else:
                                other_t.grad = other_t.grad + out.grad
                        return

                    grad_out_np = np.asarray(out.grad, dtype=np.float32)
                    if self.requires_grad:
                        grad_self = _sum_to_shape(grad_out_np, self.shape)
                        self.grad = grad_self if self.grad is None else (self.grad + grad_self)
                    if other_t.requires_grad:
                        grad_other = _sum_to_shape(grad_out_np, other_t.shape)
                        other_t.grad = grad_other if other_t.grad is None else (other_t.grad + grad_other)

            out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self.__add__(other)

    # Subtraction
    def __sub__(self, other: ArrayLike) -> "Tensor":
        if _is_number(other):
            return self.__add__(-float(other))
        other_t = self._ensure_tensor(other)
        # Limited broadcasting (PyTorch-like): (rows, cols) - (cols,)
        if len(self.shape) == 2 and len(other_t.shape) == 1 and other_t.shape[0] == self.shape[1]:
            return self.add_rowvec(-other_t)
        return self.__add__(-other_t)

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        if _is_number(other):
            # scalar - tensor
            return (-self).__add__(float(other))
        return self._ensure_tensor(other).__sub__(self)

    # True division
    def __truediv__(self, other: ArrayLike) -> "Tensor":
        if _is_number(other):
            s = float(other)
            track = is_grad_enabled() and self.requires_grad
            if self.device == "gpu":
                return self * (1.0 / s)
            out = Tensor(self.data / s, requires_grad=track, _children=(self,) if track else None, _op="/scalar", device=self.device)
            if track:
                out._prev = {self}
                def _backward() -> None:
                    if out.grad is None:
                        return
                    grad_self = out.grad / s
                    if self.grad is None:
                        self.grad = grad_self
                    else:
                        self.grad = self.grad + grad_self
                out._backward = _backward
            return out

        other_t = self._ensure_tensor(other)
        track = is_grad_enabled() and (self.requires_grad or other_t.requires_grad)
        # For GPU, use reciprocal-based division: a / b = a * (1/b)
        if self.device == "gpu" or other_t.device == "gpu":
            # Compute 1/other on GPU using mul_scalar if possible, or elementwise
            # For now use efficient pattern: a * other^(-1)
            inv_other_np = 1.0 / np.asarray(other_t.numpy(), dtype=np.float32)
            inv_other_buf = vk.to_gpu(inv_other_np) if (self.device == "gpu" or other_t.device == "gpu") else None
            if inv_other_buf is not None:
                vk_out = vk.mul(self._as_vkbuf(), inv_other_buf)
                vk.free(inv_other_buf)
                out = Tensor._from_vkbuf(vk_out, requires_grad=track)
                out._op = "/"
            else:
                out = Tensor(self.data / other_t.data, requires_grad=track, _children=(self, other_t) if track else None, _op="/", device=self.device)
        else:
            out = Tensor(self.data / other_t.data, requires_grad=track, _children=(self, other_t) if track else None, _op="/", device=self.device)

        if track:
            out._prev = {self, other_t}

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    if self.shape != other_t.shape:
                        grad_out_np = vk.to_cpu(out.grad_vkbuf)
                        a_np = np.asarray(self.numpy(), dtype=np.float32)
                        b_np = np.asarray(other_t.numpy(), dtype=np.float32)
                        if self.requires_grad:
                            grad_self_full = grad_out_np / np.broadcast_to(b_np, grad_out_np.shape)
                            grad_self = _sum_to_shape(grad_self_full, self.shape)
                            self._accum_grad_vk(vk.to_gpu(np.asarray(grad_self, dtype=np.float32)))
                        if other_t.requires_grad:
                            grad_other_full = -grad_out_np * np.broadcast_to(a_np, grad_out_np.shape) / (np.broadcast_to(b_np, grad_out_np.shape) ** 2)
                            grad_other = _sum_to_shape(grad_other_full, other_t.shape)
                            other_t._accum_grad_vk(vk.to_gpu(np.asarray(grad_other, dtype=np.float32)))
                        return
                    # grad_self = grad_out / other (can reuse as multiplication by reciprocal)
                    if self.requires_grad:
                        inv_other_np = 1.0 / np.asarray(other_t.numpy(), dtype=np.float32)
                        inv_other_buf = vk.to_gpu(inv_other_np)
                        grad_a = vk.mul(out.grad_vkbuf, inv_other_buf)
                        vk.free(inv_other_buf)
                        self._accum_grad_vk(grad_a)
                    # grad_other = -grad_out * self / (other ** 2)
                    if other_t.requires_grad:
                        grad_out_np = vk.to_cpu(out.grad_vkbuf)
                        grad_other_np = -grad_out_np * np.asarray(self.numpy(), dtype=np.float32) / (np.asarray(other_t.numpy(), dtype=np.float32) ** 2)
                        grad_b_buf = vk.to_gpu(np.asarray(grad_other_np, dtype=np.float32))
                        other_t._accum_grad_vk(grad_b_buf)
                else:
                    if out.grad is None:
                        return
                    grad_out_np = np.asarray(out.grad, dtype=np.float32)
                    a_np = np.asarray(self.data, dtype=np.float32)
                    b_np = np.asarray(other_t.data, dtype=np.float32)
                    if self.requires_grad:
                        grad_self_full = grad_out_np / np.broadcast_to(b_np, grad_out_np.shape)
                        grad_self = _sum_to_shape(grad_self_full, self.shape)
                        self.grad = grad_self if self.grad is None else (self.grad + grad_self)
                    if other_t.requires_grad:
                        grad_other_full = -grad_out_np * np.broadcast_to(a_np, grad_out_np.shape) / (np.broadcast_to(b_np, grad_out_np.shape) ** 2)
                        grad_other = _sum_to_shape(grad_other_full, other_t.shape)
                        other_t.grad = grad_other if other_t.grad is None else (other_t.grad + grad_other)

            out._backward = _backward
        return out

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return self._ensure_tensor(other).__truediv__(self)

    # Negation
    def __neg__(self) -> "Tensor":
        track = is_grad_enabled() and self.requires_grad
        if self.device == "gpu":
            vk_out = vk.neg(self._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "neg"
        else:
            out = Tensor(-self.data, requires_grad=track, _children=(self,) if track else None, _op="neg", device=self.device)

        if track:
            out._prev = {self}

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    self._accum_grad_vk(vk.neg(out.grad_vkbuf))
                else:
                    if out.grad is None:
                        return
                    if self.grad is None:
                        self.grad = -out.grad.copy()
                    else:
                        self.grad = self.grad - out.grad

            out._backward = _backward
        return out

    # Multiplication (elementwise)
    def __mul__(self, other: ArrayLike) -> "Tensor":
        if _is_number(other):
            s = float(other)
            track = is_grad_enabled() and self.requires_grad
            if self.device == "gpu":
                vk_out = vk.mul_scalar(self._as_vkbuf(), s)
                out = Tensor._from_vkbuf(vk_out, requires_grad=track)
                out._op = "*scalar"
            else:
                out = Tensor(self.data * s, requires_grad=track, _children=(self,) if track else None, _op="*scalar", device=self.device)

            if track:
                out._prev = {self}

                def _backward() -> None:
                    if out.device == "gpu":
                        if out.grad_vkbuf is None:
                            return
                        self._accum_grad_vk(vk.mul_scalar(out.grad_vkbuf, s))
                    else:
                        if out.grad is None:
                            return
                        grad_self = out.grad * s
                        self.grad = grad_self if self.grad is None else (self.grad + grad_self)

                out._backward = _backward
            return out

        other_t = self._ensure_tensor(other)
        # Limited broadcasting (PyTorch-like): (rows, cols) * (cols,)
        # Route through mul_rowvec so gradients for the row-vector are correct.
        if len(self.shape) == 2 and len(other_t.shape) == 1 and other_t.shape[0] == self.shape[1]:
            return self.mul_rowvec(other_t)

        track = is_grad_enabled() and (self.requires_grad or other_t.requires_grad)
        if self.device == "gpu" or other_t.device == "gpu":
            vk_out = vk.mul(self._as_vkbuf(), other_t._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "*"
        else:
            out_data = self.data * other_t.data
            out = Tensor(out_data, requires_grad=track, _children=(self, other_t) if track else None, _op="*", device=self.device)

        if track:
            out._prev = {self, other_t}

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    if self.shape == other_t.shape:
                        if self.requires_grad:
                            self._accum_grad_vk(vk.mul(other_t._as_vkbuf(), out.grad_vkbuf))
                        if other_t.requires_grad:
                            other_t._accum_grad_vk(vk.mul(self._as_vkbuf(), out.grad_vkbuf))
                        return

                    # Broadcasted case: compute full grad on GPU then reduce-to-shape on GPU.
                    if self.requires_grad:
                        grad_full = vk.mul(other_t._as_vkbuf(), out.grad_vkbuf)
                        try:
                            self._accum_grad_vk(vk.sum_to_shape(grad_full, self.shape))
                        finally:
                            vk.free(grad_full)
                    if other_t.requires_grad:
                        grad_full = vk.mul(self._as_vkbuf(), out.grad_vkbuf)
                        try:
                            other_t._accum_grad_vk(vk.sum_to_shape(grad_full, other_t.shape))
                        finally:
                            vk.free(grad_full)
                else:
                    if out.grad is None:
                        return
                    if self.shape == other_t.shape:
                        if self.requires_grad:
                            grad_self = other_t.data * out.grad
                            if self.grad is None:
                                self.grad = grad_self
                            else:
                                self.grad = self.grad + grad_self
                        if other_t.requires_grad:
                            grad_other = self.data * out.grad
                            if other_t.grad is None:
                                other_t.grad = grad_other
                            else:
                                other_t.grad = other_t.grad + grad_other
                        return

                    grad_out_np = np.asarray(out.grad, dtype=np.float32)
                    a_np = np.asarray(self.data, dtype=np.float32)
                    b_np = np.asarray(other_t.data, dtype=np.float32)
                    if self.requires_grad:
                        grad_self_full = np.broadcast_to(b_np, grad_out_np.shape) * grad_out_np
                        grad_self = _sum_to_shape(grad_self_full, self.shape)
                        self.grad = grad_self if self.grad is None else (self.grad + grad_self)
                    if other_t.requires_grad:
                        grad_other_full = np.broadcast_to(a_np, grad_out_np.shape) * grad_out_np
                        grad_other = _sum_to_shape(grad_other_full, other_t.shape)
                        other_t.grad = grad_other if other_t.grad is None else (other_t.grad + grad_other)

            out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self.__mul__(other)

    # Matrix multiplication
    def __matmul__(self, other: ArrayLike) -> "Tensor":
        other_t = self._ensure_tensor(other)
        track = is_grad_enabled() and (self.requires_grad or other_t.requires_grad)
        if self.device == "gpu" or other_t.device == "gpu":
            vk_out = vk.matmul(self._as_vkbuf(), other_t._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "matmul"
        else:
            out_data = self.data @ other_t.data
            out = Tensor(out_data, requires_grad=track, _children=(self, other_t) if track else None, _op="matmul", device=self.device)

        if track:
            out._prev = {self, other_t}

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    # dA = dC @ B^T
                    # dB = A^T @ dC
                    if self.requires_grad:
                        b_t = vk.transpose2d(other_t._as_vkbuf())
                        self._accum_grad_vk(vk.matmul(out.grad_vkbuf, b_t))
                        vk.free(b_t)
                    if other_t.requires_grad:
                        a_t = vk.transpose2d(self._as_vkbuf())
                        other_t._accum_grad_vk(vk.matmul(a_t, out.grad_vkbuf))
                        vk.free(a_t)
                else:
                    if out.grad is None:
                        return
                    if self.requires_grad:
                        grad_self = out.grad @ other_t.data.T
                        if self.grad is None:
                            self.grad = grad_self
                        else:
                            self.grad = self.grad + grad_self
                    if other_t.requires_grad:
                        grad_other = self.data.T @ out.grad
                        if other_t.grad is None:
                            other_t.grad = grad_other
                        else:
                            other_t.grad = other_t.grad + grad_other

            out._backward = _backward
        return out

    # Reductions
    def sum(self, axis: int | tuple[int, ...] | None = None) -> "Tensor":
        track = is_grad_enabled() and self.requires_grad
        if self.device == "gpu":
            axes = _normalize_axes(axis, len(self.shape))
            if axis is None:
                vk_out = vk.reduce_sum(self._as_vkbuf())
                out = Tensor._from_vkbuf(vk_out, requires_grad=track)
                out._op = "sum"
            elif len(axes) == 1 and len(self.shape) == 2 and axes[0] in (0, 1):
                a = self._as_vkbuf()
                tmp_t: "vk.VulkanBuffer | None" = None
                try:
                    if axes[0] == 0:
                        vk_out = vk.reduce_sum_rows(a)
                    else:
                        tmp_t = vk.transpose2d(a)
                        vk_out = vk.reduce_sum_rows(tmp_t)
                    out = Tensor._from_vkbuf(vk_out, requires_grad=track)
                    out._op = "sum_dim"
                finally:
                    if tmp_t is not None:
                        vk.free(tmp_t)
            else:
                out = Tensor(
                    self.numpy().sum(axis=axis),
                    requires_grad=track,
                    _children=(self,) if track else None,
                    _op="sum_dim",
                    device=self.device,
                )
        else:
            out = Tensor(
                self.numpy().sum(axis=axis),
                requires_grad=track,
                _children=(self,) if track else None,
                _op="sum" if axis is None else "sum_dim",
                device=self.device,
            )

        if track:
            out._prev = {self}
            axes = _normalize_axes(axis, len(self.shape))

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    if axis is None:
                        out_buf = vk.empty(self.shape)
                        self._accum_grad_vk(vk.scale_fill(out.grad_vkbuf, out_buf, 1.0))
                        return
                    # GPU-optimized path for 2D axis=0/1 reductions
                    if len(axes) == 1 and len(self.shape) == 2 and axes[0] in (0, 1):
                        n, m = self.shape
                        grad_buf = out.grad_vkbuf
                        if axes[0] == 0:
                            # Sum over rows: grad shape is (M,), need to broadcast to (N, M)
                            # Use add_rowvec to add the gradient row-vector to a zero matrix
                            zeros = vk.empty((n, m))
                            grad_broadcast_buf = vk.add_rowvec(zeros, grad_buf)
                            self._accum_grad_vk(grad_broadcast_buf)
                            vk.free(zeros)
                        else:
                            # Sum over cols: grad shape is (N,), need to broadcast to (N, M)
                            # Transpose, use add_rowvec, then transpose back
                            grad_t = vk.transpose2d(grad_buf)
                            zeros = vk.empty((m, n))
                            grad_broadcast_t = vk.add_rowvec(zeros, grad_t)
                            grad_broadcast_buf = vk.transpose2d(grad_broadcast_t)
                            self._accum_grad_vk(grad_broadcast_buf)
                            vk.free(grad_t)
                            vk.free(zeros)
                            vk.free(grad_broadcast_t)
                        return
                    # General GPU path: reshape grad to insert singleton dims, then broadcast on GPU.
                    expanded_shape = list(out.shape)
                    for ax in sorted(axes):
                        expanded_shape.insert(ax, 1)
                    grad_view = vk.view(out.grad_vkbuf, tuple(expanded_shape))
                    try:
                        grad_broadcast_buf = vk.broadcast_to(grad_view, self.shape)
                        self._accum_grad_vk(grad_broadcast_buf)
                    finally:
                        vk.free(grad_view)
                    return
                else:
                    if out.grad is None:
                        return
                    grad_out = out.grad

                if axis is None:
                    grad_broadcast = np.ones_like(self.numpy(), dtype=np.float32) * grad_out
                else:
                    expanded = grad_out
                    for ax in sorted(axes):
                        expanded = np.expand_dims(expanded, axis=ax)
                    grad_broadcast = np.broadcast_to(expanded, self.shape).astype(np.float32, copy=False)
                if self.device == "gpu":
                    self._accum_grad_vk(vk.to_gpu(np.asarray(grad_broadcast, dtype=np.float32)))
                else:
                    self.grad = grad_broadcast if self.grad is None else (self.grad + grad_broadcast)

            out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None) -> "Tensor":
        track = is_grad_enabled() and self.requires_grad
        if self.device == "gpu":
            axes = _normalize_axes(axis, len(self.shape))
            if axis is None:
                vk_out = vk.mean(self._as_vkbuf())
                out = Tensor._from_vkbuf(vk_out, requires_grad=track)
                out._op = "mean"
            elif len(axes) == 1 and len(self.shape) == 2 and axes[0] in (0, 1):
                rows, cols = self.shape
                denom = float(rows if axes[0] == 0 else cols)
                a = self._as_vkbuf()
                tmp_t: "vk.VulkanBuffer | None" = None
                tmp_sum: "vk.VulkanBuffer | None" = None
                try:
                    if axes[0] == 0:
                        tmp_sum = vk.reduce_sum_rows(a)
                    else:
                        tmp_t = vk.transpose2d(a)
                        tmp_sum = vk.reduce_sum_rows(tmp_t)
                    vk_out = vk.mul_scalar(tmp_sum, 1.0 / max(1.0, denom))
                    out = Tensor._from_vkbuf(vk_out, requires_grad=track)
                    out._op = "mean_dim"
                finally:
                    if tmp_sum is not None:
                        vk.free(tmp_sum)
                    if tmp_t is not None:
                        vk.free(tmp_t)
            else:
                out = Tensor(
                    self.numpy().mean(axis=axis),
                    requires_grad=track,
                    _children=(self,) if track else None,
                    _op="mean_dim",
                    device=self.device,
                )
        else:
            out = Tensor(
                self.numpy().mean(axis=axis),
                requires_grad=track,
                _children=(self,) if track else None,
                _op="mean" if axis is None else "mean_dim",
                device=self.device,
            )

        if track:
            out._prev = {self}
            axes = _normalize_axes(axis, len(self.shape))
            denom = float(np.prod([self.shape[ax] for ax in axes]))

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    if axis is None:
                        n = float(np.prod(self.shape))
                        out_buf = vk.empty(self.shape)
                        self._accum_grad_vk(vk.scale_fill(out.grad_vkbuf, out_buf, 1.0 / max(1.0, n)))
                        return
                    # GPU-optimized path for 2D axis=0/1 reductions
                    if len(axes) == 1 and len(self.shape) == 2 and axes[0] in (0, 1):
                        n, m = self.shape
                        grad_buf = out.grad_vkbuf
                        rows, cols = self.shape
                        axis_denom = float(rows if axes[0] == 0 else cols)
                        if axes[0] == 0:
                            # Mean over rows: grad shape is (M,), need to broadcast to (N, M)
                            # Use add_rowvec to add the gradient row-vector to a zero matrix
                            scalar_grad = vk.mul_scalar(grad_buf, 1.0 / max(1.0, axis_denom))
                            zeros = vk.empty((n, m))
                            grad_broadcast_buf = vk.add_rowvec(zeros, scalar_grad)
                            self._accum_grad_vk(grad_broadcast_buf)
                            vk.free(zeros)
                            vk.free(scalar_grad)
                        else:
                            # Mean over cols: grad shape is (N,), need to broadcast to (N, M)
                            # Transpose, use add_rowvec, then transpose back
                            scalar_grad = vk.mul_scalar(grad_buf, 1.0 / max(1.0, axis_denom))
                            grad_t = vk.transpose2d(scalar_grad)
                            zeros = vk.empty((m, n))
                            grad_broadcast_t = vk.add_rowvec(zeros, grad_t)
                            grad_broadcast_buf = vk.transpose2d(grad_broadcast_t)
                            self._accum_grad_vk(grad_broadcast_buf)
                            vk.free(grad_t)
                            vk.free(zeros)
                            vk.free(grad_broadcast_t)
                            vk.free(scalar_grad)
                        return
                    # General GPU path: scale, reshape to insert singleton dims, then broadcast on GPU.
                    scaled = vk.mul_scalar(out.grad_vkbuf, 1.0 / max(1.0, denom))
                    expanded_shape = list(out.shape)
                    for ax in sorted(axes):
                        expanded_shape.insert(ax, 1)
                    grad_view = vk.view(scaled, tuple(expanded_shape))
                    try:
                        grad_broadcast_buf = vk.broadcast_to(grad_view, self.shape)
                        self._accum_grad_vk(grad_broadcast_buf)
                    finally:
                        vk.free(grad_view)
                        vk.free(scaled)
                    return
                else:
                    if out.grad is None:
                        return
                    grad_out = out.grad

                if axis is None:
                    grad_broadcast = np.ones_like(self.numpy(), dtype=np.float32) * grad_out / self.data.size
                else:
                    expanded = grad_out
                    for ax in sorted(axes):
                        expanded = np.expand_dims(expanded, axis=ax)
                    grad_broadcast = np.broadcast_to(expanded, self.shape).astype(np.float32, copy=False) / max(1.0, denom)
                if self.device == "gpu":
                    self._accum_grad_vk(vk.to_gpu(np.asarray(grad_broadcast, dtype=np.float32)))
                else:
                    self.grad = grad_broadcast if self.grad is None else (self.grad + grad_broadcast)

            out._backward = _backward
        return out

    def max(self, axis: int | None = None) -> "Tensor":
        x_np = np.asarray(self.numpy(), dtype=np.float32)
        result = x_np.max(axis=axis)
        track = is_grad_enabled() and self.requires_grad
        out = Tensor(result, requires_grad=track, _children=(self,) if track else None, _op="max", device=self.device)

        if track:
            out._prev = {self}
            if axis is None:
                mask = (x_np == result).astype(np.float32)
                divisor = float(mask.sum())
            else:
                expanded = np.expand_dims(result, axis=axis)
                mask = (x_np == expanded).astype(np.float32)
                divisor = np.maximum(mask.sum(axis=axis, keepdims=True), 1.0)

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    grad_out = vk.to_cpu(out.grad_vkbuf)
                else:
                    if out.grad is None:
                        return
                    grad_out = out.grad

                if axis is None:
                    grad_in = mask * (np.asarray(grad_out, dtype=np.float32) / max(1.0, float(divisor)))
                else:
                    grad_in = mask * np.expand_dims(np.asarray(grad_out, dtype=np.float32), axis=axis) / divisor
                if self.device == "gpu":
                    self._accum_grad_vk(vk.to_gpu(np.asarray(grad_in, dtype=np.float32)))
                else:
                    self.grad = grad_in if self.grad is None else (self.grad + grad_in)

            out._backward = _backward
        return out

    def min(self, axis: int | None = None) -> "Tensor":
        x_np = np.asarray(self.numpy(), dtype=np.float32)
        result = x_np.min(axis=axis)
        track = is_grad_enabled() and self.requires_grad
        out = Tensor(result, requires_grad=track, _children=(self,) if track else None, _op="min", device=self.device)

        if track:
            out._prev = {self}
            if axis is None:
                mask = (x_np == result).astype(np.float32)
                divisor = float(mask.sum())
            else:
                expanded = np.expand_dims(result, axis=axis)
                mask = (x_np == expanded).astype(np.float32)
                divisor = np.maximum(mask.sum(axis=axis, keepdims=True), 1.0)

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    grad_out = vk.to_cpu(out.grad_vkbuf)
                else:
                    if out.grad is None:
                        return
                    grad_out = out.grad

                if axis is None:
                    grad_in = mask * (np.asarray(grad_out, dtype=np.float32) / max(1.0, float(divisor)))
                else:
                    grad_in = mask * np.expand_dims(np.asarray(grad_out, dtype=np.float32), axis=axis) / divisor
                if self.device == "gpu":
                    self._accum_grad_vk(vk.to_gpu(np.asarray(grad_in, dtype=np.float32)))
                else:
                    self.grad = grad_in if self.grad is None else (self.grad + grad_in)

            out._backward = _backward
        return out

    def argmax(self, axis: int | None = None):
        result = np.argmax(np.asarray(self.numpy(), dtype=np.float32), axis=axis)
        if axis is None:
            return int(result)
        return np.asarray(result, dtype=np.int64)

    def argmin(self, axis: int | None = None):
        result = np.argmin(np.asarray(self.numpy(), dtype=np.float32), axis=axis)
        if axis is None:
            return int(result)
        return np.asarray(result, dtype=np.int64)

    def _unary_numpy_op(
        self,
        *,
        op: str,
        forward: Callable[[np.ndarray], np.ndarray],
        backward: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> "Tensor":
        x_np = self.numpy()
        y_np = np.asarray(forward(x_np), dtype=np.float32)
        track = is_grad_enabled() and self.requires_grad

        # Cache x and y on GPU for efficient backward if on GPU
        x_gpu_buf: "vk.VulkanBuffer | None" = None
        y_gpu_buf: "vk.VulkanBuffer | None" = None

        if self.device == "gpu":
            y_gpu_buf = vk.to_gpu(y_np)
            x_gpu_buf = self._as_vkbuf()
            out = Tensor._from_vkbuf(y_gpu_buf, requires_grad=track)
            out._op = op
        else:
            out = Tensor(
                y_np,
                requires_grad=track,
                _children=(self,) if track else None,
                _op=op,
                device=self.device,
            )

        if track:
            out._prev = {self}

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    # GPU path: minimize CPU transfers by computing backward on smaller gradient tensor
                    grad_out = vk.to_cpu(out.grad_vkbuf)
                    # For GPU backward, keep x and y cached
                    grad_in = np.asarray(backward(x_np, y_np, grad_out), dtype=np.float32)
                    self._accum_grad_vk(vk.to_gpu(grad_in))
                else:
                    if out.grad is None:
                        return
                    grad_in = np.asarray(backward(x_np, y_np, out.grad), dtype=np.float32)
                    self.grad = grad_in if self.grad is None else (self.grad + grad_in)

            out._backward = _backward

        return out

    # Non-linearities
    def relu(self) -> "Tensor":
        track = is_grad_enabled() and self.requires_grad
        if self.device == "gpu":
            vk_out = vk.relu(self._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "relu"
        else:
            out_data = np.maximum(self.data, 0)
            out = Tensor(out_data, requires_grad=track, _children=(self,) if track else None, _op="relu", device=self.device)

        if track:
            out._prev = {self}

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    self._accum_grad_vk(vk.relu_backward(out.grad_vkbuf, self._as_vkbuf()))
                else:
                    if out.grad is None:
                        return
                    grad = out.grad * (self.data > 0)
                    if self.grad is None:
                        self.grad = grad
                    else:
                        self.grad = self.grad + grad

            out._backward = _backward
        return out

    # Other activations
    def sigmoid(self) -> "Tensor":
        track = is_grad_enabled() and self.requires_grad
        
        if self.device == "gpu":
            # GPU forward: y = 1 / (1 + exp(-x))
            x_vkbuf = self._as_vkbuf()
            x_np = self.numpy()
            y_np = 1.0 / (1.0 + np.exp(-x_np))
            y_vkbuf = vk.to_gpu(y_np)
            out = Tensor._from_vkbuf(y_vkbuf, requires_grad=track)
            out._op = "sigmoid"
            
            if track:
                out._prev = {self}
                
                def _backward() -> None:
                    if out.grad_vkbuf is None:
                        return
                    # GPU backward: grad = grad_out * y * (1 - y)
                    # Step 1: 1 - y = -y + 1
                    neg_y = vk.neg(y_vkbuf)
                    one_minus_y = vk.add_scalar(neg_y, 1.0)
                    # Step 2: grad = grad_out * y * (1 - y)
                    grad = vk.mul(vk.mul(out.grad_vkbuf, y_vkbuf), one_minus_y)
                    self._accum_grad_vk(grad)
                
                out._backward = _backward
            return out
        
        # CPU path
        def _forward(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-x))

        def _backward(_: np.ndarray, y: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
            return grad_out * y * (1.0 - y)

        return self._unary_numpy_op(op="sigmoid", forward=_forward, backward=_backward)

    def tanh(self) -> "Tensor":
        track = is_grad_enabled() and self.requires_grad
        
        if self.device == "gpu":
            # GPU forward: y = tanh(x)
            x_vkbuf = self._as_vkbuf()
            x_np = self.numpy()
            y_np = np.tanh(x_np)
            y_vkbuf = vk.to_gpu(y_np)
            out = Tensor._from_vkbuf(y_vkbuf, requires_grad=track)
            out._op = "tanh"
            
            if track:
                out._prev = {self}
                
                def _backward() -> None:
                    if out.grad_vkbuf is None:
                        return
                    # GPU backward: grad = grad_out * (1 - y^2)
                    # Step 1: y^2
                    y_squared = vk.mul(y_vkbuf, y_vkbuf)
                    # Step 2: 1 - y^2 = -y^2 + 1
                    neg_y_sq = vk.neg(y_squared)
                    one_minus_y_sq = vk.add_scalar(neg_y_sq, 1.0)
                    # Step 3: grad = grad_out * (1 - y^2)
                    grad = vk.mul(out.grad_vkbuf, one_minus_y_sq)
                    self._accum_grad_vk(grad)
                
                out._backward = _backward
            return out
        
        # CPU path
        def _forward(x: np.ndarray) -> np.ndarray:
            return np.tanh(x)

        def _backward(_: np.ndarray, y: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
            return grad_out * (1.0 - y ** 2)

        return self._unary_numpy_op(op="tanh", forward=_forward, backward=_backward)

    def gelu(self) -> "Tensor":
        track = is_grad_enabled() and self.requires_grad
        if self.device == "gpu":
            vk_out = vk.gelu(self._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "gelu"

            if track:
                out._prev = {self}

                def _backward() -> None:
                    if out.grad_vkbuf is None:
                        return
                    self._accum_grad_vk(vk.gelu_backward(out.grad_vkbuf, self._as_vkbuf()))

                out._backward = _backward
            return out

        sqrt_2_over_pi = np.float32(np.sqrt(2.0 / np.pi))

        def _forward(x: np.ndarray) -> np.ndarray:
            inner = sqrt_2_over_pi * (x + np.float32(0.044715) * (x ** 3))
            return np.float32(0.5) * x * (1.0 + np.tanh(inner))

        def _backward(x: np.ndarray, _: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
            inner = sqrt_2_over_pi * (x + np.float32(0.044715) * (x ** 3))
            tanh_inner = np.tanh(inner)
            sech2 = 1.0 - tanh_inner ** 2
            inner_grad = sqrt_2_over_pi * (1.0 + np.float32(0.134145) * (x ** 2))
            grad = np.float32(0.5) * (1.0 + tanh_inner) + np.float32(0.5) * x * sech2 * inner_grad
            return grad_out * grad

        return self._unary_numpy_op(op="gelu", forward=_forward, backward=_backward)

    def silu(self) -> "Tensor":
        track = is_grad_enabled() and self.requires_grad
        if self.device == "gpu":
            vk_out = vk.silu(self._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "silu"

            if track:
                out._prev = {self}

                def _backward() -> None:
                    if out.grad_vkbuf is None:
                        return
                    self._accum_grad_vk(vk.silu_backward(out.grad_vkbuf, self._as_vkbuf()))

                out._backward = _backward
            return out

        def _forward(x: np.ndarray) -> np.ndarray:
            sig = 1.0 / (1.0 + np.exp(-x))
            return x * sig

        def _backward(x: np.ndarray, _: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
            sig = 1.0 / (1.0 + np.exp(-x))
            grad = sig * (1.0 + x * (1.0 - sig))
            return grad_out * grad

        return self._unary_numpy_op(op="silu", forward=_forward, backward=_backward)

    def leaky_relu(self, alpha: float = 0.01) -> "Tensor":
        alpha = float(alpha)

        track = is_grad_enabled() and self.requires_grad
        if self.device == "gpu":
            vk_out = vk.leaky_relu(self._as_vkbuf(), alpha)
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "leaky_relu"

            if track:
                out._prev = {self}

                def _backward() -> None:
                    if out.grad_vkbuf is None:
                        return
                    self._accum_grad_vk(vk.leaky_relu_backward(out.grad_vkbuf, self._as_vkbuf(), alpha))

                out._backward = _backward
            return out

        def _forward(x: np.ndarray) -> np.ndarray:
            return np.where(x > 0.0, x, alpha * x)

        def _backward(x: np.ndarray, _: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
            return grad_out * np.where(x > 0.0, 1.0, alpha)

        return self._unary_numpy_op(op="leaky_relu", forward=_forward, backward=_backward)

    def elu(self, alpha: float = 1.0) -> "Tensor":
        alpha = float(alpha)

        track = is_grad_enabled() and self.requires_grad
        if self.device == "gpu":
            vk_out = vk.elu(self._as_vkbuf(), alpha)
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "elu"

            if track:
                out._prev = {self}

                def _backward() -> None:
                    if out.grad_vkbuf is None:
                        return
                    self._accum_grad_vk(vk.elu_backward(out.grad_vkbuf, self._as_vkbuf(), alpha))

                out._backward = _backward
            return out

        def _forward(x: np.ndarray) -> np.ndarray:
            return np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0))

        def _backward(x: np.ndarray, y: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
            grad = np.where(x > 0.0, 1.0, y + alpha)
            return grad_out * grad

        return self._unary_numpy_op(op="elu", forward=_forward, backward=_backward)

    def abs(self) -> "Tensor":
        def _forward(x: np.ndarray) -> np.ndarray:
            return np.abs(x)

        def _backward(x: np.ndarray, _: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
            return grad_out * np.sign(x)

        return self._unary_numpy_op(op="abs", forward=_forward, backward=_backward)

    def clamp(self, min_value: float | None = None, max_value: float | None = None) -> "Tensor":
        if min_value is None and max_value is None:
            return self

        def _forward(x: np.ndarray) -> np.ndarray:
            lower = -np.inf if min_value is None else float(min_value)
            upper = np.inf if max_value is None else float(max_value)
            return np.clip(x, lower, upper)

        def _backward(x: np.ndarray, _: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
            mask = np.ones_like(x, dtype=np.float32)
            if min_value is not None:
                mask = mask * (x >= float(min_value))
            if max_value is not None:
                mask = mask * (x <= float(max_value))
            return grad_out * mask

        return self._unary_numpy_op(op="clamp", forward=_forward, backward=_backward)

    def __getitem__(self, index) -> "Tensor":
        x_np = np.asarray(self.numpy(), dtype=np.float32)
        out_np = np.asarray(x_np[index], dtype=np.float32)
        track = is_grad_enabled() and self.requires_grad
        out = Tensor(out_np, requires_grad=track, _children=(self,) if track else None, _op="getitem", device=self.device)

        if track:
            out._prev = {self}

            def _backward() -> None:
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    grad_out = vk.to_cpu(out.grad_vkbuf)
                else:
                    if out.grad is None:
                        return
                    grad_out = out.grad

                grad_in = np.zeros_like(x_np, dtype=np.float32)
                np.add.at(grad_in, index, np.asarray(grad_out, dtype=np.float32))
                if self.device == "gpu":
                    self._accum_grad_vk(vk.to_gpu(grad_in))
                else:
                    self.grad = grad_in if self.grad is None else (self.grad + grad_in)

            out._backward = _backward
        return out

    def split(self, split_size_or_sections: int | Sequence[int], dim: int = 0) -> tuple["Tensor", ...]:
        axis = _normalize_dim(dim, len(self.shape))
        size = self.shape[axis]
        if isinstance(split_size_or_sections, int):
            split_size = int(split_size_or_sections)
            if split_size <= 0:
                raise ValueError("split_size must be > 0")
            sections = []
            start = 0
            while start < size:
                stop = min(size, start + split_size)
                sections.append(stop - start)
                start = stop
        else:
            sections = [int(s) for s in split_size_or_sections]
            if sum(sections) != size:
                raise ValueError(f"split sections {sections} do not sum to dimension size {size}")
            for sec in sections:
                if sec < 0:
                    raise ValueError("split sections must be >= 0")

        outputs: list[Tensor] = []
        if self.device == "gpu":
            track = is_grad_enabled() and self.requires_grad
            start = 0
            vk_parts = vk.split(self._as_vkbuf(), sections, dim=axis)
            for sec, start_idx, part in zip(sections, np.cumsum([0] + sections[:-1]).tolist(), vk_parts):
                out = Tensor._from_vkbuf(part, requires_grad=track)
                out._op = "split"
                if track:
                    out._prev = {self}

                    def _backward(out=out, sec=sec, start_idx=start_idx) -> None:
                        if out.grad_vkbuf is None:
                            return
                        self._accum_grad_vk(vk.scatter_slice(out.grad_vkbuf, self.shape, axis, start_idx))

                    out._backward = _backward
                outputs.append(out)
            return tuple(outputs)

        start = 0
        for sec in sections:
            sl = [slice(None)] * len(self.shape)
            sl[axis] = slice(start, start + sec)
            outputs.append(self[tuple(sl)])
            start += sec
        return tuple(outputs)

    def chunk(self, chunks: int, dim: int = 0) -> tuple["Tensor", ...]:
        chunks = int(chunks)
        if chunks <= 0:
            raise ValueError("chunks must be > 0")
        axis = _normalize_dim(dim, len(self.shape))
        size = self.shape[axis]
        split_size = int(np.ceil(size / chunks)) if size > 0 else 1
        return self.split(split_size, dim=axis)

    # Utilities
    def numpy(self) -> np.ndarray:
        """Return the underlying NumPy array.

        Note: For GPU tensors this triggers a device->host readback.
        """
        if self._vkbuf is not None:
            # Download a CPU copy for inspection, but keep the tensor on GPU.
            self.data = vk.to_cpu(self._vkbuf)
        return self.data

    def free(self) -> None:
        """Free GPU buffers owned by this tensor (no-op for CPU tensors).

        rasptorch does not currently have automatic GPU memory reclamation; demos and
        benchmarks should call this to avoid leaks.
        """
        if self.grad_vkbuf is not None:
            vk.free(self.grad_vkbuf)
            self.grad_vkbuf = None
        if self._vkbuf is not None:
            vk.free(self._vkbuf)
            self._vkbuf = None

    def add_rowvec(self, b: "Tensor") -> "Tensor":
        """Broadcast-add a 1D row vector to a 2D matrix (GPU-optimized)."""

        track = is_grad_enabled() and (self.requires_grad or b.requires_grad)

        if self.device == "gpu" or b.device == "gpu":
            vk_out = vk.add_rowvec(self._as_vkbuf(), b._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "add_rowvec"

            if track:
                out._prev = {self, b}

                def _backward() -> None:
                    if out.grad_vkbuf is None:
                        return
                    if self.requires_grad:
                        self._accum_grad_vk(out.grad_vkbuf)
                    if b.requires_grad:
                        b._accum_grad_vk(vk.reduce_sum_rows(out.grad_vkbuf))

                out._backward = _backward
            return out

        # CPU fallback
        out_data = self.data + b.data
        out = Tensor(out_data, requires_grad=track, _children=(self, b) if track else None, _op="add_rowvec", device=self.device)

        if track:
            def _backward() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    self.grad = out.grad.copy() if self.grad is None else (self.grad + out.grad)
                if b.requires_grad:
                    gb = out.grad.sum(axis=0)
                    b.grad = gb if b.grad is None else (b.grad + gb)

            out._backward = _backward
        return out

    def mul_rowvec(self, b: "Tensor") -> "Tensor":
        """Broadcast-multiply a 1D row vector with a 2D matrix (CPU+GPU)."""

        track = is_grad_enabled() and (self.requires_grad or b.requires_grad)

        if self.device == "gpu" or b.device == "gpu":
            vk_out = vk.mul_rowvec(self._as_vkbuf(), b._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=track)
            out._op = "mul_rowvec"

            if track:
                out._prev = {self, b}

                def _backward() -> None:
                    if out.grad_vkbuf is None:
                        return
                    if self.requires_grad:
                        self._accum_grad_vk(vk.mul_rowvec(out.grad_vkbuf, b._as_vkbuf()))
                    if b.requires_grad:
                        tmp = vk.mul(out.grad_vkbuf, self._as_vkbuf())
                        gb = vk.reduce_sum_rows(tmp)
                        vk.free(tmp)
                        b._accum_grad_vk(gb)

                out._backward = _backward
            return out

        # CPU fallback
        out_data = self.data * b.data
        out = Tensor(out_data, requires_grad=track, _children=(self, b) if track else None, _op="mul_rowvec", device=self.device)

        if track:
            def _backward() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    ga = out.grad * b.data
                    self.grad = ga if self.grad is None else (self.grad + ga)
                if b.requires_grad:
                    gb = (out.grad * self.data).sum(axis=0)
                    b.grad = gb if b.grad is None else (b.grad + gb)

            out._backward = _backward
        return out

    # Device handling
    def to(self, device: str) -> "Tensor":
        """Return a copy of this Tensor on the given device."""
        if device == self.device:
            return self

        if device == "cpu":
            # Stay in NumPy – data is already on CPU.
            if self._vkbuf is not None:
                return Tensor(vk.to_cpu(self._vkbuf), requires_grad=False, device="cpu")
            return Tensor(self.data.copy(), requires_grad=self.requires_grad, device="cpu")

        if device == "gpu":
            # Upload once and keep the VulkanBuffer so subsequent ops avoid
            # per-op upload/download.
            if self._vkbuf is not None:
                return self
            return Tensor._from_vkbuf(vk.to_gpu(self.data), requires_grad=self.requires_grad)

        raise NotImplementedError(f"Unknown device: {device}")

    def half(self) -> "Tensor":
        quantized = np.asarray(self.numpy(), dtype=np.float16).astype(np.float32)
        out = Tensor(quantized, requires_grad=self.requires_grad, device="cpu")
        return out.to(self.device)

    def float(self) -> "Tensor":
        out = Tensor(np.asarray(self.numpy(), dtype=np.float32), requires_grad=self.requires_grad, device="cpu")
        return out.to(self.device)


class Parameter(Tensor):
    """A trainable tensor, like torch.nn.Parameter."""

    def __init__(self, data: ArrayLike, *, device: str = "cpu", _vkbuf: "vk.VulkanBuffer | None" = None) -> None:
        super().__init__(data, requires_grad=True, device=device, _vkbuf=_vkbuf)

    def to(self, device: str) -> "Parameter":
        if device == self.device:
            return self
        if device == "cpu":
            if self._vkbuf is not None:
                return Parameter(vk.to_cpu(self._vkbuf), device="cpu")
            return Parameter(self.data.copy(), device="cpu")
        if device == "gpu":
            if self._vkbuf is not None:
                return self
            return Parameter(np.empty(self.data.shape, dtype=np.float32), device="gpu", _vkbuf=vk.to_gpu(self.data))
        raise NotImplementedError(f"Unknown device: {device}")


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    if not tensors:
        raise ValueError("cat expects at least one tensor")
    first = tensors[0]
    ndim = len(first.shape)
    axis = _normalize_dim(dim, ndim)
    base_shape = list(first.shape)
    for t in tensors[1:]:
        if len(t.shape) != ndim:
            raise ValueError("all tensors must have the same rank for cat")
        for i, (a, b) in enumerate(zip(base_shape, t.shape)):
            if i != axis and a != b:
                raise ValueError(f"cat dimension mismatch at dim {i}: {a} != {b}")

    device = "gpu" if any(t.device == "gpu" for t in tensors) else first.device
    track = is_grad_enabled() and any(t.requires_grad for t in tensors)
    if device == "gpu":
        out = Tensor._from_vkbuf(vk.concat([t._as_vkbuf() for t in tensors], dim=axis), requires_grad=track)
        out._op = "cat"
    else:
        arrays = [np.asarray(t.numpy(), dtype=np.float32) for t in tensors]
        out_np = np.concatenate(arrays, axis=axis)
        out = Tensor(out_np, requires_grad=track, _children=tuple(tensors) if track else None, _op="cat", device=device)

    if track:
        out._prev = set(tensors)
        sizes = [t.shape[axis] for t in tensors]

        def _backward() -> None:
            if out.device == "gpu":
                if out.grad_vkbuf is None:
                    return
                pieces = vk.split(out.grad_vkbuf, [int(s) for s in sizes], dim=axis)
                for t, piece in zip(tensors, pieces):
                    if t.device == "gpu":
                        t._accum_grad_vk(piece)
                    else:
                        grad_piece = vk.to_cpu(piece)
                        t.grad = grad_piece if t.grad is None else (t.grad + grad_piece)
                return
            else:
                if out.grad is None:
                    return
                grad_out = out.grad

            start = 0
            for t, size in zip(tensors, sizes):
                sl = [slice(None)] * grad_out.ndim
                sl[axis] = slice(start, start + size)
                grad_piece = np.asarray(grad_out[tuple(sl)], dtype=np.float32)
                if t.device == "gpu":
                    t._accum_grad_vk(vk.to_gpu(grad_piece))
                else:
                    t.grad = grad_piece if t.grad is None else (t.grad + grad_piece)
                start += size

        out._backward = _backward
    return out


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    if not tensors:
        raise ValueError("stack expects at least one tensor")
    first_shape = tensors[0].shape
    for t in tensors[1:]:
        if t.shape != first_shape:
            raise ValueError(f"stack expects equal shapes, got {first_shape} and {t.shape}")
    axis = _normalize_dim(dim, len(first_shape) + 1, allow_end=True)
    return cat([t.unsqueeze(axis) for t in tensors], dim=axis)
