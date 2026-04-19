from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .backend import connect_backend, get_backend
from . import vulkan_backend as vk
from .tensor import Tensor


def backend_device(backend_name: str | None) -> str:
    """Map a backend name to a coarse device class."""
    name = str(backend_name or "").strip().lower()
    return "gpu" if name in {"vulkan", "opencl", "cuda"} else "cpu"


def backend_device_label(backend_name: str | None) -> str:
    """Map a backend name to user-facing device text."""
    name = str(backend_name or "").strip().lower()
    if name == "vulkan":
        return "gpu (Vulkan)"
    if name == "opencl":
        return "gpu (OpenCL)"
    if name == "cuda":
        return "gpu (CUDA)"
    return "cpu"


def resolve_device(device: str | None) -> str:
    """Resolve a user-provided device string.

    Accepted values:
    - "cpu": force CPU
    - "gpu": request GPU (may still fall back internally if Vulkan is unavailable)
    - "auto": use GPU if Vulkan initializes successfully, else CPU

    Returns: "cpu" or "gpu".
    """

    if device is None:
        return "cpu"

    d = str(device).strip().lower()
    if d == "auto":
        chosen = connect_backend("vulkan", strict=False)
        return backend_device(chosen.name)
    if d in ("cpu", "gpu"):
        # For GPU, proactively try init so callers can surface failures early.
        if d == "gpu":
            chosen = connect_backend("vulkan", strict=False)
            if chosen.name != "vulkan":
                return "cpu"
        else:
            connect_backend("cpu", strict=False)
        return d

    raise ValueError(f"Unknown device: {device!r} (expected 'cpu', 'gpu', or 'auto')")


def resolve_backend(backend: str | None) -> str:
    """Resolve and connect a backend selection.

    Accepted values:
    - "auto": prefer Vulkan, otherwise CPU
    - "cpu" | "vulkan" | "opencl" | "cuda"

    Returns the active backend name after connection.
    """

    if backend is None:
        return get_backend().name

    b = str(backend).strip().lower()
    if b == "auto":
        return connect_backend("vulkan", strict=False).name
    if b == "numpy":
        b = "cpu"
    if b in {"cpu", "vulkan", "opencl", "cuda"}:
        return connect_backend(b, strict=False).name
    raise ValueError(f"Unknown backend: {backend!r} (expected 'auto', 'numpy', 'vulkan', 'opencl', or 'cuda')")


def _materialize_params(parameters: Iterable[Tensor]) -> list[Tensor]:
    return [p for p in parameters if getattr(p, "requires_grad", False)]


def _accum_grad(param: Tensor, grad: np.ndarray) -> None:
    grad_np = np.asarray(grad, dtype=np.float32)
    if param.device == "gpu":
        param._accum_grad_vk(vk.to_gpu(grad_np))
    else:
        param.grad = grad_np if param.grad is None else (param.grad + grad_np)


def _penalty_tensor(value: float, device: str, requires_grad: bool, op: str) -> Tensor:
    arr = np.array([value], dtype=np.float32)
    if device == "gpu":
        out = Tensor._from_vkbuf(vk.to_gpu(arr), requires_grad=requires_grad)
        out._op = op
        return out
    return Tensor(arr, requires_grad=requires_grad, device="cpu", _op=op)


def _scalar_grad_value(t: Tensor) -> float:
    if t.device == "gpu":
        if t.grad_vkbuf is None:
            return 0.0
        return float(vk.to_cpu(t.grad_vkbuf).reshape(-1)[0])
    if t.grad is None:
        return 0.0
    return float(np.asarray(t.grad, dtype=np.float32).reshape(-1)[0])


def clip_grad_norm_(parameters: Iterable[Tensor], max_norm: float, norm_type: float = 2.0) -> float:
    params = [p for p in _materialize_params(parameters) if p.grad is not None or p.grad_vkbuf is not None]
    if not params:
        return 0.0

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if max_norm < 0.0:
        raise ValueError("max_norm must be >= 0")

    if np.isinf(norm_type):
        total_norm = max(
            float(np.max(np.abs(vk.to_cpu(p.grad_vkbuf) if p.grad_vkbuf is not None else p.grad))) for p in params
        )
    else:
        total = 0.0
        for p in params:
            grad = vk.to_cpu(p.grad_vkbuf) if p.grad_vkbuf is not None else p.grad
            assert grad is not None
            total += float(np.sum(np.abs(grad) ** norm_type))
        total_norm = total ** (1.0 / norm_type)

    if total_norm == 0.0 or total_norm <= max_norm:
        return total_norm

    scale = max_norm / (total_norm + 1e-12)
    for p in params:
        if p.grad_vkbuf is not None:
            scaled = vk.mul_scalar(p.grad_vkbuf, scale)
            vk.free(p.grad_vkbuf)
            p.grad_vkbuf = scaled
        else:
            assert p.grad is not None
            p.grad = p.grad * scale

    return total_norm


def clip_grad_value_(parameters: Iterable[Tensor], clip_value: float) -> None:
    params = [p for p in _materialize_params(parameters) if p.grad is not None or p.grad_vkbuf is not None]
    limit = abs(float(clip_value))
    for p in params:
        if p.grad_vkbuf is not None:
            grad = np.clip(vk.to_cpu(p.grad_vkbuf), -limit, limit).astype(np.float32)
            vk.write(p.grad_vkbuf, grad)
        else:
            assert p.grad is not None
            p.grad = np.clip(p.grad, -limit, limit).astype(np.float32)


def l1_regularization(model, lambda_: float) -> Tensor:
    params = _materialize_params(model.parameters())
    device = "gpu" if any(p.device == "gpu" for p in params) else "cpu"
    value = float(lambda_) * sum(float(np.abs(p.numpy()).sum()) for p in params)
    out = _penalty_tensor(value, device=device, requires_grad=bool(params), op="l1_regularization")

    if params:
        out._prev = set(params)

        def _backward() -> None:
            scale = _scalar_grad_value(out)
            if scale == 0.0:
                return
            for p in params:
                _accum_grad(p, scale * float(lambda_) * np.sign(p.numpy()))

        out._backward = _backward

    return out


def l2_regularization(model, lambda_: float) -> Tensor:
    params = _materialize_params(model.parameters())
    device = "gpu" if any(p.device == "gpu" for p in params) else "cpu"
    value = float(lambda_) * sum(float((p.numpy() ** 2).sum()) for p in params)
    out = _penalty_tensor(value, device=device, requires_grad=bool(params), op="l2_regularization")

    if params:
        out._prev = set(params)

        def _backward() -> None:
            scale = _scalar_grad_value(out)
            if scale == 0.0:
                return
            for p in params:
                _accum_grad(p, scale * 2.0 * float(lambda_) * p.numpy())

        out._backward = _backward

    return out


def total_variation_loss(tensor: Tensor) -> Tensor:
    x = np.asarray(tensor.numpy(), dtype=np.float32)
    if x.ndim == 0:
        return _penalty_tensor(0.0, device=tensor.device, requires_grad=tensor.requires_grad, op="total_variation_loss")

    value = 0.0
    for axis in range(x.ndim):
        head = [slice(None)] * x.ndim
        tail = [slice(None)] * x.ndim
        head[axis] = slice(1, None)
        tail[axis] = slice(None, -1)
        value += float(np.abs(x[tuple(head)] - x[tuple(tail)]).sum())

    out = _penalty_tensor(value, device=tensor.device, requires_grad=tensor.requires_grad, op="total_variation_loss")

    if tensor.requires_grad:
        out._prev = {tensor}

        def _backward() -> None:
            scale = _scalar_grad_value(out)
            if scale == 0.0:
                return
            grad = np.zeros_like(x, dtype=np.float32)
            for axis in range(x.ndim):
                head = [slice(None)] * x.ndim
                tail = [slice(None)] * x.ndim
                head[axis] = slice(1, None)
                tail[axis] = slice(None, -1)
                diff = x[tuple(head)] - x[tuple(tail)]
                sign = np.sign(diff).astype(np.float32)
                grad[tuple(head)] += sign
                grad[tuple(tail)] -= sign
            _accum_grad(tensor, scale * grad)

        out._backward = _backward

    return out
