from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Set

import numpy as np

from . import vulkan_backend as vk


ArrayLike = Any


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer))


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

    @property
    def shape(self) -> tuple[int, ...]:
        if self._vkbuf is not None:
            return self._vkbuf.shape
        return self.data.shape

    @property
    def T(self) -> "Tensor":
        """Transpose view (2D only for now)."""
        if self.device == "gpu":
            vk_out = vk.transpose2d(self._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad)
            out._prev = {self}
            out._op = "transpose"
        else:
            out = Tensor(self.data.T, _children=(self,), _op="transpose", device=self.device)

        def _backward() -> None:
            if not self.requires_grad:
                return
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
        out.requires_grad = self.requires_grad
        return out

    def view(self, *shape: int) -> "Tensor":
        """Reshape tensor without changing underlying storage (like torch.view)."""

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])  # type: ignore[assignment]
        new_shape = tuple(int(s) for s in shape)
        if int(np.prod(new_shape)) != int(np.prod(self.shape)):
            raise ValueError(f"view cannot change number of elements: {self.shape} -> {new_shape}")

        if self.device == "gpu":
            vk_out = vk.view(self._as_vkbuf(), new_shape)
            out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad)
            out._prev = {self}
            out._op = "view"
        else:
            out = Tensor(self.data.reshape(new_shape), _children=(self,), _op="view", device=self.device)

        def _backward() -> None:
            if not self.requires_grad:
                return
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
        out.requires_grad = self.requires_grad
        return out

    def reshape(self, *shape: int) -> "Tensor":
        return self.view(*shape)

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
            if self.device == "gpu":
                vk_out = vk.add_scalar(self._as_vkbuf(), s)
                out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad)
                out._prev = {self}
                out._op = "+scalar"
            else:
                out = Tensor(self.data + s, _children=(self,), _op="+scalar", device=self.device)

            def _backward() -> None:
                if not self.requires_grad:
                    return
                if out.device == "gpu":
                    if out.grad_vkbuf is None:
                        return
                    self._accum_grad_vk(out.grad_vkbuf)
                else:
                    if out.grad is None:
                        return
                    self.grad = out.grad.copy() if self.grad is None else (self.grad + out.grad)

            out._backward = _backward
            out.requires_grad = self.requires_grad
            return out

        other_t = self._ensure_tensor(other)
        # Limited broadcasting (PyTorch-like): (rows, cols) + (cols,)
        # Route through add_rowvec so gradients for the row-vector are correct.
        if len(self.shape) == 2 and len(other_t.shape) == 1 and other_t.shape[0] == self.shape[1]:
            return self.add_rowvec(other_t)
        if self.device == "gpu" or other_t.device == "gpu":
            vk_out = vk.add(self._as_vkbuf(), other_t._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad or other_t.requires_grad)
            out._prev = {self, other_t}
            out._op = "+"
        else:
            out_data = self.data + other_t.data
            out = Tensor(out_data, _children=(self, other_t), _op="+", device=self.device)

        def _backward() -> None:
            if out.device == "gpu":
                if out.grad_vkbuf is None:
                    return
                if self.requires_grad:
                    self._accum_grad_vk(out.grad_vkbuf)
                if other_t.requires_grad:
                    other_t._accum_grad_vk(out.grad_vkbuf)
            else:
                if out.grad is None:
                    return
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

        out._backward = _backward
        out.requires_grad = self.requires_grad or other_t.requires_grad
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
            if self.device == "gpu":
                return self * (1.0 / s)

        other_t = self._ensure_tensor(other)
        out = Tensor(self.data / other_t.data, _children=(self, other_t), _op="/")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad / other_t.data
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad = self.grad + grad_self
            if other_t.requires_grad:
                grad_other = -out.grad * self.data / (other_t.data ** 2)
                if other_t.grad is None:
                    other_t.grad = grad_other
                else:
                    other_t.grad = other_t.grad + grad_other

        out._backward = _backward
        out.requires_grad = self.requires_grad or other_t.requires_grad
        return out

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return self._ensure_tensor(other).__truediv__(self)

    # Negation
    def __neg__(self) -> "Tensor":
        if self.device == "gpu":
            vk_out = vk.neg(self._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad)
            out._prev = {self}
            out._op = "neg"
        else:
            out = Tensor(-self.data, _children=(self,), _op="neg", device=self.device)

        def _backward() -> None:
            if not self.requires_grad:
                return
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
        out.requires_grad = self.requires_grad
        return out

    # Multiplication (elementwise)
    def __mul__(self, other: ArrayLike) -> "Tensor":
        if _is_number(other):
            s = float(other)
            if self.device == "gpu":
                vk_out = vk.mul_scalar(self._as_vkbuf(), s)
                out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad)
                out._prev = {self}
                out._op = "*scalar"
            else:
                out = Tensor(self.data * s, _children=(self,), _op="*scalar", device=self.device)

            def _backward() -> None:
                if not self.requires_grad:
                    return
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
            out.requires_grad = self.requires_grad
            return out

        other_t = self._ensure_tensor(other)
        # Limited broadcasting (PyTorch-like): (rows, cols) * (cols,)
        # Route through mul_rowvec so gradients for the row-vector are correct.
        if len(self.shape) == 2 and len(other_t.shape) == 1 and other_t.shape[0] == self.shape[1]:
            return self.mul_rowvec(other_t)
        if self.device == "gpu" or other_t.device == "gpu":
            vk_out = vk.mul(self._as_vkbuf(), other_t._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad or other_t.requires_grad)
            out._prev = {self, other_t}
            out._op = "*"
        else:
            out_data = self.data * other_t.data
            out = Tensor(out_data, _children=(self, other_t), _op="*", device=self.device)

        def _backward() -> None:
            if out.device == "gpu":
                if out.grad_vkbuf is None:
                    return
                if self.requires_grad:
                    self._accum_grad_vk(vk.mul(other_t._as_vkbuf(), out.grad_vkbuf))
                if other_t.requires_grad:
                    other_t._accum_grad_vk(vk.mul(self._as_vkbuf(), out.grad_vkbuf))
            else:
                if out.grad is None:
                    return
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

        out._backward = _backward
        out.requires_grad = self.requires_grad or other_t.requires_grad
        return out

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self.__mul__(other)

    # Matrix multiplication
    def __matmul__(self, other: ArrayLike) -> "Tensor":
        other_t = self._ensure_tensor(other)
        if self.device == "gpu" or other_t.device == "gpu":
            vk_out = vk.matmul(self._as_vkbuf(), other_t._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad or other_t.requires_grad)
            out._prev = {self, other_t}
            out._op = "matmul"
        else:
            out_data = self.data @ other_t.data
            out = Tensor(out_data, _children=(self, other_t), _op="matmul", device=self.device)

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
        out.requires_grad = self.requires_grad or other_t.requires_grad
        return out

    # Reductions
    def sum(self) -> "Tensor":
        if self.device == "gpu":
            vk_out = vk.reduce_sum(self._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad)
            out._prev = {self}
            out._op = "sum"
        else:
            out = Tensor(self.data.sum(), _children=(self,), _op="sum", device=self.device)

        def _backward() -> None:
            if not self.requires_grad:
                return
            if out.device == "gpu":
                if out.grad_vkbuf is None:
                    return
                # Broadcast scalar grad to input shape.
                out_buf = vk.empty(self.shape)
                self._accum_grad_vk(vk.scale_fill(out.grad_vkbuf, out_buf, 1.0))
            else:
                if out.grad is None:
                    return
                grad_broadcast = np.ones_like(self.data, dtype=self.data.dtype) * out.grad
                if self.grad is None:
                    self.grad = grad_broadcast
                else:
                    self.grad = self.grad + grad_broadcast

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def mean(self) -> "Tensor":
        if self.device == "gpu":
            vk_out = vk.mean(self._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad)
            out._prev = {self}
            out._op = "mean"
        else:
            out = Tensor(self.data.mean(), _children=(self,), _op="mean", device=self.device)

        def _backward() -> None:
            if not self.requires_grad:
                return
            if out.device == "gpu":
                if out.grad_vkbuf is None:
                    return
                n = float(np.prod(self.shape))
                out_buf = vk.empty(self.shape)
                self._accum_grad_vk(vk.scale_fill(out.grad_vkbuf, out_buf, 1.0 / max(1.0, n)))
            else:
                if out.grad is None:
                    return
                grad_broadcast = (
                    np.ones_like(self.data, dtype=self.data.dtype) * out.grad / self.data.size
                )
                if self.grad is None:
                    self.grad = grad_broadcast
                else:
                    self.grad = self.grad + grad_broadcast

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # Non-linearities
    def relu(self) -> "Tensor":
        if self.device == "gpu":
            vk_out = vk.relu(self._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad)
            out._prev = {self}
            out._op = "relu"
        else:
            out_data = np.maximum(self.data, 0)
            out = Tensor(out_data, _children=(self,), _op="relu", device=self.device)

        def _backward() -> None:
            if not self.requires_grad:
                return
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
        out.requires_grad = self.requires_grad
        return out

    # Other activations
    def sigmoid(self) -> "Tensor":
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig, _children=(self,), _op="sigmoid", device=self.device)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad * sig * (1 - sig)
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,), _op="tanh", device=self.device)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad * (1 - t ** 2)
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

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

        if self.device == "gpu" or b.device == "gpu":
            vk_out = vk.add_rowvec(self._as_vkbuf(), b._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad or b.requires_grad)
            out._prev = {self, b}
            out._op = "add_rowvec"

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
        out = Tensor(out_data, _children=(self, b), _op="add_rowvec", device=self.device)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = out.grad.copy() if self.grad is None else (self.grad + out.grad)
            if b.requires_grad:
                gb = out.grad.sum(axis=0)
                b.grad = gb if b.grad is None else (b.grad + gb)

        out._backward = _backward
        out.requires_grad = self.requires_grad or b.requires_grad
        return out

    def mul_rowvec(self, b: "Tensor") -> "Tensor":
        """Broadcast-multiply a 1D row vector with a 2D matrix (CPU+GPU)."""

        if self.device == "gpu" or b.device == "gpu":
            vk_out = vk.mul_rowvec(self._as_vkbuf(), b._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out, requires_grad=self.requires_grad or b.requires_grad)
            out._prev = {self, b}
            out._op = "mul_rowvec"

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
        out = Tensor(out_data, _children=(self, b), _op="mul_rowvec", device=self.device)

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
        out.requires_grad = self.requires_grad or b.requires_grad
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
