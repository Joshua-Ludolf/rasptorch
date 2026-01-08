from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Set

import numpy as np

from . import vulkan_backend as vk


ArrayLike = Any


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
            # GPU autograd isn't implemented yet.
            requires_grad = False
            device = "gpu"
        else:
            self.data: np.ndarray = np.array(data, dtype=np.float32)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad: bool = requires_grad
        self.device: str = device
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set(_children) if _children is not None else set()
        self._op: str = _op

    @classmethod
    def _from_vkbuf(cls, buf: "vk.VulkanBuffer") -> "Tensor":
        return cls(
            np.empty(buf.shape, dtype=np.float32),
            requires_grad=False,
            device="gpu",
            _vkbuf=buf,
        )

    def _as_vkbuf(self) -> "vk.VulkanBuffer":
        if self._vkbuf is not None:
            return self._vkbuf
        return vk.to_gpu(self.data)

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
        out = Tensor(self.data.T, _children=(self,), _op="transpose", device=self.device)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad.T
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # ------------------------------------------------------------------
    # Autograd core
    # ------------------------------------------------------------------
    def backward(self, grad: ArrayLike | None = None) -> None:
        if self.device == "gpu":
            raise NotImplementedError(
                "Autograd on device='gpu' is not implemented yet. "
                "Call .to('cpu') before backward()."
            )
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar outputs")
            grad_arr = np.ones_like(self.data, dtype=self.data.dtype)
        else:
            grad_arr = np.array(grad, dtype=self.data.dtype)

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
        other_t = self._ensure_tensor(other)
        if self.device == "gpu" or other_t.device == "gpu":
            vk_out = vk.add(self._as_vkbuf(), other_t._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out)
        else:
            out_data = self.data + other_t.data
            out = Tensor(out_data, _children=(self, other_t), _op="+", device=self.device)

        def _backward() -> None:
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
        return self.__add__(-self._ensure_tensor(other))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return self._ensure_tensor(other).__sub__(self)

    # True division
    def __truediv__(self, other: ArrayLike) -> "Tensor":
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
        out = Tensor(-self.data, _children=(self,), _op="neg", device=self.device)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
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
        other_t = self._ensure_tensor(other)
        if self.device == "gpu" or other_t.device == "gpu":
            vk_out = vk.mul(self._as_vkbuf(), other_t._as_vkbuf())
            out = Tensor._from_vkbuf(vk_out)
        else:
            out_data = self.data * other_t.data
            out = Tensor(out_data, _children=(self, other_t), _op="*", device=self.device)

        def _backward() -> None:
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
            out = Tensor._from_vkbuf(vk_out)
        else:
            out_data = self.data @ other_t.data
            out = Tensor(out_data, _children=(self, other_t), _op="matmul", device=self.device)

        def _backward() -> None:
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
        out = Tensor(self.data.sum(), _children=(self,), _op="sum", device=self.device)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
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
        out = Tensor(self.data.mean(), _children=(self,), _op="mean", device=self.device)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
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
            out = Tensor._from_vkbuf(vk_out)
        else:
            out_data = np.maximum(self.data, 0)
            out = Tensor(out_data, _children=(self,), _op="relu", device=self.device)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
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
        """Return the underlying NumPy array (view)."""
        if self._vkbuf is not None:
            # Download a CPU copy for inspection, but keep the tensor on GPU.
            self.data = vk.to_cpu(self._vkbuf)
        return self.data

    # Device handling
    def to(self, device: str) -> "Tensor":
        """Return a copy of this Tensor on the given device.

        Currently only "cpu" is implemented; other devices are placeholders
        for future Raspberry Pi GPU experimentation.
        """
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
            return Tensor._from_vkbuf(vk.to_gpu(self.data))

        raise NotImplementedError(f"Unknown device: {device}")


class Parameter(Tensor):
    """A trainable tensor, like torch.nn.Parameter."""

    def __init__(self, data: ArrayLike) -> None:
        super().__init__(data, requires_grad=True)
