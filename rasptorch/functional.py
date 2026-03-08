from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from . import vulkan_backend as vk
from .tensor import Tensor, is_grad_enabled


def _canonical_dim(dim: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("ndim must be > 0")
    d = int(dim)
    if d < 0:
        d += int(ndim)
    if d < 0 or d >= int(ndim):
        raise ValueError(f"dim out of range for ndim={ndim}: dim={dim}")
    return d


def _make_output_tensor(value: np.ndarray, *, device: str, requires_grad: bool, op: str) -> Tensor:
    arr = np.asarray(value, dtype=np.float32)
    if device == "gpu":
        out = Tensor._from_vkbuf(vk.to_gpu(arr), requires_grad=requires_grad)
        out._op = op
        return out
    return Tensor(arr, requires_grad=requires_grad, _op=op, device="cpu")


def _loss_scalar_tensor(value: float, *, device: str, requires_grad: bool, op: str) -> Tensor:
    return _make_output_tensor(np.array([value], dtype=np.float32), device=device, requires_grad=requires_grad, op=op)


def _scalar_grad_value(t: Tensor) -> float:
    if t.device == "gpu":
        if t.grad_vkbuf is None:
            return 0.0
        return float(vk.to_cpu(t.grad_vkbuf).reshape(-1)[0])
    if t.grad is None:
        return 0.0
    return float(np.asarray(t.grad, dtype=np.float32).reshape(-1)[0])


def _accum_tensor_grad(t: Tensor, grad: np.ndarray) -> None:
    grad_np = np.asarray(grad, dtype=np.float32)
    if not t.requires_grad:
        return
    if t.device == "gpu":
        t._accum_grad_vk(vk.to_gpu(grad_np))
    else:
        t.grad = grad_np if t.grad is None else (t.grad + grad_np)


def _target_distribution(target: Tensor | Sequence[int] | np.ndarray, num_classes: int) -> np.ndarray:
    if isinstance(target, Tensor):
        target_np = target.numpy()
    else:
        target_np = np.asarray(target)

    if target_np.ndim == 1:
        labels = target_np.astype(np.int64, copy=False).reshape(-1)
        if labels.size == 0:
            return np.zeros((0, num_classes), dtype=np.float32)
        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError(f"target indices out of range [0,{num_classes})")
        out = np.zeros((labels.size, num_classes), dtype=np.float32)
        out[np.arange(labels.size), labels] = 1.0
        return out

    if target_np.shape != (target_np.shape[0], num_classes):
        raise ValueError(f"expected target shape [N,{num_classes}], got {target_np.shape}")
    return np.asarray(target_np, dtype=np.float32)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax over a dimension (CPU autograd).

    Notes:
    - CPU path supports autograd.
    - GPU path supports autograd for 2D row-wise softmax (dim=-1/1).
    """

    d = _canonical_dim(dim, len(x.shape))
    track = is_grad_enabled() and x.requires_grad

    if x.device == "gpu":
        # GPU path currently supports 2D row-wise softmax only.
        if len(x.shape) != 2:
            raise NotImplementedError("GPU softmax currently supports 2D tensors only")
        if d not in (1,):
            raise NotImplementedError("GPU softmax currently supports dim=-1/1 only")

        vk_out = vk.softmax2d(x._as_vkbuf())
        out = Tensor._from_vkbuf(vk_out, requires_grad=track)
        out._op = "softmax"

        if track:
            out._prev = {x}

            def _backward() -> None:
                if out.grad_vkbuf is None:
                    return
                dx = vk.softmax2d_backward(out._as_vkbuf(), out.grad_vkbuf)
                x._accum_grad_vk(dx)

            out._backward = _backward

        return out

    x_np = x.numpy()
    x_max = x_np.max(axis=d, keepdims=True)
    z = x_np - x_max
    e = np.exp(z)
    s = e / e.sum(axis=d, keepdims=True)

    out = Tensor(s.astype(np.float32, copy=False), requires_grad=track, _children=(x,) if track else None, _op="softmax", device="cpu")

    if track:
        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad
            # dx = y * (g - sum(g*y))
            gy = (g * s).sum(axis=d, keepdims=True)
            dx = s * (g - gy)
            x.grad = dx if x.grad is None else (x.grad + dx)

        out._backward = _backward

    return out


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """LogSoftmax over a dimension (CPU autograd).

    Notes:
    - CPU path supports autograd.
    - GPU path supports autograd for 2D row-wise log_softmax (dim=-1/1).
    """

    d = _canonical_dim(dim, len(x.shape))
    track = is_grad_enabled() and x.requires_grad

    if x.device == "gpu":
        # GPU path currently supports 2D row-wise log_softmax only.
        if len(x.shape) != 2:
            raise NotImplementedError("GPU log_softmax currently supports 2D tensors only")
        if d not in (1,):
            raise NotImplementedError("GPU log_softmax currently supports dim=-1/1 only")

        vk_out = vk.log_softmax2d(x._as_vkbuf())
        out = Tensor._from_vkbuf(vk_out, requires_grad=track)
        out._op = "log_softmax"

        if track:
            out._prev = {x}

            def _backward() -> None:
                if out.grad_vkbuf is None:
                    return
                dx = vk.log_softmax2d_backward(out._as_vkbuf(), out.grad_vkbuf)
                x._accum_grad_vk(dx)

            out._backward = _backward

        return out

    x_np = x.numpy()
    x_max = x_np.max(axis=d, keepdims=True)
    z = x_np - x_max
    lse = np.log(np.exp(z).sum(axis=d, keepdims=True)) + x_max
    out_np = x_np - lse

    out = Tensor(out_np.astype(np.float32, copy=False), requires_grad=track, _children=(x,) if track else None, _op="log_softmax", device="cpu")

    if track:
        # softmax = exp(log_softmax)
        s = np.exp(out_np)

        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad
            # dx = g - softmax * sum(g)
            sum_g = g.sum(axis=d, keepdims=True)
            dx = g - s * sum_g
            x.grad = dx if x.grad is None else (x.grad + dx)

        out._backward = _backward

    return out


def relu(x: Tensor) -> Tensor:
    return x.relu()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def silu(x: Tensor) -> Tensor:
    return x.silu()


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    return x.leaky_relu(alpha)


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    return x.elu(alpha)


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    diff = input - target
    return (diff * diff).mean()


def binary_cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    if input.shape != target.shape:
        raise ValueError(f"binary_cross_entropy expects matching shapes, got {input.shape} and {target.shape}")

    x = np.asarray(input.numpy(), dtype=np.float32)
    t = np.asarray(target.numpy(), dtype=np.float32)
    clipped = np.clip(x, 1e-7, 1.0 - 1e-7)
    loss = -np.mean(t * np.log(clipped) + (1.0 - t) * np.log(1.0 - clipped))

    requires_grad = is_grad_enabled() and input.requires_grad
    out = _loss_scalar_tensor(loss, device=input.device, requires_grad=requires_grad, op="binary_cross_entropy")

    if requires_grad:
        out._prev = {input, target}

        def _backward() -> None:
            scale = _scalar_grad_value(out)
            if scale == 0.0:
                return
            grad = ((clipped - t) / np.maximum(clipped * (1.0 - clipped), 1e-7)) / max(1, clipped.size)
            _accum_tensor_grad(input, scale * grad)

        out._backward = _backward

    return out


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor) -> Tensor:
    if input.shape != target.shape:
        raise ValueError(
            f"binary_cross_entropy_with_logits expects matching shapes, got {input.shape} and {target.shape}"
        )

    x = np.asarray(input.numpy(), dtype=np.float32)
    t = np.asarray(target.numpy(), dtype=np.float32)
    loss_np = np.maximum(x, 0.0) - x * t + np.log1p(np.exp(-np.abs(x)))
    loss = float(loss_np.mean())

    requires_grad = is_grad_enabled() and input.requires_grad
    out = _loss_scalar_tensor(loss, device=input.device, requires_grad=requires_grad, op="binary_cross_entropy_with_logits")

    if requires_grad:
        out._prev = {input, target}

        def _backward() -> None:
            scale = _scalar_grad_value(out)
            if scale == 0.0:
                return
            sig = 1.0 / (1.0 + np.exp(-x))
            grad = (sig - t) / max(1, x.size)
            _accum_tensor_grad(input, scale * grad)

        out._backward = _backward

    return out


def one_hot(labels: Tensor | Sequence[int] | np.ndarray, num_classes: int) -> Tensor:
    """Create a float32 one-hot matrix of shape [N, C].

    Notes:
    - This is a small convenience helper for demos/tests.
    - Output is a CPU tensor by default; call `.to('gpu')` if needed.
    """

    if isinstance(labels, Tensor):
        lab = labels.numpy().astype(np.int64).reshape(-1)
    else:
        lab = np.asarray(labels, dtype=np.int64).reshape(-1)

    if num_classes <= 0:
        raise ValueError("num_classes must be > 0")
    if lab.size == 0:
        return Tensor(np.zeros((0, int(num_classes)), dtype=np.float32))
    if lab.min() < 0 or lab.max() >= num_classes:
        raise ValueError(f"labels out of range [0,{num_classes})")

    out = np.zeros((int(lab.size), int(num_classes)), dtype=np.float32)
    out[np.arange(lab.size), lab] = 1.0
    return Tensor(out)


def cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    """Softmax cross-entropy loss (mean reduction).

    Expects:
    - logits: [N, C]
    - target: [N, C] (one-hot or probabilities)
    """

    if logits.shape != target.shape or len(logits.shape) != 2:
        raise ValueError(f"cross_entropy expects logits/target both [N,C]; got {logits.shape} and {target.shape}")

    requires_grad = is_grad_enabled() and logits.requires_grad

    # GPU path: forward uses a per-sample loss kernel + mean reduction;
    # backward uses the dedicated (softmax - target)/N kernel.
    if logits.device == "gpu" or target.device == "gpu":
        if logits.device != "gpu" or target.device != "gpu":
            raise ValueError("cross_entropy GPU path requires both logits and target on 'gpu'")

        logits_g = logits
        target_g = target

        loss_vec = vk.softmax_xent_loss_vec(logits_g._as_vkbuf(), target_g._as_vkbuf())
        try:
            loss_buf = vk.mean(loss_vec)
        finally:
            vk.free(loss_vec)

        out = Tensor._from_vkbuf(loss_buf, requires_grad=requires_grad)
        out._op = "cross_entropy"

        if requires_grad:
            out._prev = {logits_g, target_g}

            def _backward() -> None:
                if out.grad_vkbuf is None:
                    return

                g = float(vk.to_cpu(out.grad_vkbuf).reshape(-1)[0])
                grad_logits = vk.softmax_xent_backward(logits_g._as_vkbuf(), target_g._as_vkbuf())
                if g != 1.0:
                    scaled = vk.mul_scalar(grad_logits, float(g))
                    vk.free(grad_logits)
                    grad_logits = scaled
                logits_g._accum_grad_vk(grad_logits)

            out._backward = _backward
        return out

    # CPU fallback
    l = logits.numpy()
    t = target.numpy()
    m = l.max(axis=1, keepdims=True)
    z = l - m
    logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True)) + m
    loss_vec = -(t * (l - logsumexp)).sum(axis=1)
    loss = float(loss_vec.mean())
    out = Tensor(
        np.array([loss], dtype=np.float32),
        requires_grad=requires_grad,
        _children=(logits, target) if requires_grad else None,
        _op="cross_entropy",
        device="cpu",
    )

    if requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            # grad logits: (softmax - target)/N
            e = np.exp(z)
            s = e / e.sum(axis=1, keepdims=True)
            g = (s - t) / max(1, l.shape[0])
            g = g * float(out.grad.reshape(-1)[0])
            logits.grad = g if logits.grad is None else (logits.grad + g)

        out._backward = _backward
    return out


def nll_loss(log_probs: Tensor, target: Tensor | Sequence[int] | np.ndarray) -> Tensor:
    if len(log_probs.shape) != 2:
        raise ValueError(f"nll_loss expects log_probs shape [N,C], got {log_probs.shape}")

    lp = np.asarray(log_probs.numpy(), dtype=np.float32)
    target_dist = _target_distribution(target, lp.shape[1])
    if target_dist.shape[0] != lp.shape[0]:
        raise ValueError(f"target batch mismatch: expected {lp.shape[0]}, got {target_dist.shape[0]}")

    loss = float(-(target_dist * lp).sum(axis=1).mean())
    requires_grad = is_grad_enabled() and log_probs.requires_grad
    out = _loss_scalar_tensor(loss, device=log_probs.device, requires_grad=requires_grad, op="nll_loss")

    if requires_grad:
        out._prev = {log_probs}

        def _backward() -> None:
            scale = _scalar_grad_value(out)
            if scale == 0.0:
                return
            grad = -target_dist / max(1, lp.shape[0])
            _accum_tensor_grad(log_probs, scale * grad)

        out._backward = _backward

    return out


def smooth_l1_loss(input: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    if input.shape != target.shape:
        raise ValueError(f"smooth_l1_loss expects matching shapes, got {input.shape} and {target.shape}")
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("beta must be > 0")

    diff = np.asarray(input.numpy(), dtype=np.float32) - np.asarray(target.numpy(), dtype=np.float32)
    abs_diff = np.abs(diff)
    loss_np = np.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
    loss = float(loss_np.mean())

    requires_grad = is_grad_enabled() and input.requires_grad
    out = _loss_scalar_tensor(loss, device=input.device, requires_grad=requires_grad, op="smooth_l1_loss")

    if requires_grad:
        out._prev = {input, target}

        def _backward() -> None:
            scale = _scalar_grad_value(out)
            if scale == 0.0:
                return
            grad = np.where(abs_diff < beta, diff / beta, np.sign(diff)) / max(1, diff.size)
            _accum_tensor_grad(input, scale * grad)

        out._backward = _backward

    return out


def label_smoothing_cross_entropy(
    logits: Tensor,
    target: Tensor | Sequence[int] | np.ndarray,
    smoothing: float = 0.1,
) -> Tensor:
    if len(logits.shape) != 2:
        raise ValueError(f"label_smoothing_cross_entropy expects logits shape [N,C], got {logits.shape}")

    smoothing = float(smoothing)
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")

    x = np.asarray(logits.numpy(), dtype=np.float32)
    num_classes = x.shape[1]
    target_dist = _target_distribution(target, num_classes)
    if target_dist.shape[0] != x.shape[0]:
        raise ValueError(f"target batch mismatch: expected {x.shape[0]}, got {target_dist.shape[0]}")

    smoothed = (1.0 - smoothing) * target_dist + smoothing / num_classes
    m = x.max(axis=1, keepdims=True)
    z = x - m
    exp_z = np.exp(z)
    softmax_np = exp_z / exp_z.sum(axis=1, keepdims=True)
    log_probs = x - (np.log(exp_z.sum(axis=1, keepdims=True)) + m)
    loss = float(-(smoothed * log_probs).sum(axis=1).mean())

    requires_grad = is_grad_enabled() and logits.requires_grad
    out = _loss_scalar_tensor(loss, device=logits.device, requires_grad=requires_grad, op="label_smoothing_cross_entropy")

    if requires_grad:
        out._prev = {logits}

        def _backward() -> None:
            scale = _scalar_grad_value(out)
            if scale == 0.0:
                return
            grad = (softmax_np - smoothed) / max(1, x.shape[0])
            _accum_tensor_grad(logits, scale * grad)

        out._backward = _backward

    return out


def cosine_similarity(a: Tensor, b: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(f"cosine_similarity expects matching shapes, got {a.shape} and {b.shape}")

    d = _canonical_dim(dim, len(a.shape))
    a_np = np.asarray(a.numpy(), dtype=np.float32)
    b_np = np.asarray(b.numpy(), dtype=np.float32)
    dot = (a_np * b_np).sum(axis=d, keepdims=True)
    a_norm = np.maximum(np.linalg.norm(a_np, axis=d, keepdims=True), eps)
    b_norm = np.maximum(np.linalg.norm(b_np, axis=d, keepdims=True), eps)
    denom = a_norm * b_norm
    out_np = (dot / denom).astype(np.float32, copy=False)
    squeezed = np.squeeze(out_np, axis=d)

    requires_grad = is_grad_enabled() and (a.requires_grad or b.requires_grad)
    device = "gpu" if a.device == "gpu" or b.device == "gpu" else "cpu"
    out = _make_output_tensor(squeezed, device=device, requires_grad=requires_grad, op="cosine_similarity")

    if requires_grad:
        out._prev = {a, b}

        def _backward() -> None:
            if out.device == "gpu":
                if out.grad_vkbuf is None:
                    return
                grad_out = vk.to_cpu(out.grad_vkbuf)
            else:
                if out.grad is None:
                    return
                grad_out = np.asarray(out.grad, dtype=np.float32)

            grad_out_expanded = np.expand_dims(grad_out, axis=d)
            if a.requires_grad:
                grad_a = grad_out_expanded * (b_np / denom - (dot * a_np) / np.maximum(a_norm ** 3 * b_norm, eps))
                _accum_tensor_grad(a, grad_a)
            if b.requires_grad:
                grad_b = grad_out_expanded * (a_np / denom - (dot * b_np) / np.maximum(b_norm ** 3 * a_norm, eps))
                _accum_tensor_grad(b, grad_b)

        out._backward = _backward

    return out
