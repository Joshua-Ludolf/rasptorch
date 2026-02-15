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


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    diff = input - target
    return (diff * diff).mean()


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
