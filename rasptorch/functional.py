from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from . import vulkan_backend as vk
from .tensor import Tensor


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

    requires_grad = logits.requires_grad

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
        out._prev = {logits_g, target_g}
        out._op = "cross_entropy"

        def _backward() -> None:
            if not logits_g.requires_grad:
                return
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
        out.requires_grad = requires_grad
        return out

    # CPU fallback
    l = logits.numpy()
    t = target.numpy()
    m = l.max(axis=1, keepdims=True)
    z = l - m
    logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True)) + m
    loss_vec = -(t * (l - logsumexp)).sum(axis=1)
    loss = float(loss_vec.mean())
    out = Tensor(np.array([loss], dtype=np.float32), _children=(logits, target), _op="cross_entropy", device="cpu")

    def _backward() -> None:
        if not logits.requires_grad:
            return
        if out.grad is None:
            return
        # grad logits: (softmax - target)/N
        e = np.exp(z)
        s = e / e.sum(axis=1, keepdims=True)
        g = (s - t) / max(1, l.shape[0])
        g = g * float(out.grad.reshape(-1)[0])
        logits.grad = g if logits.grad is None else (logits.grad + g)

    out._backward = _backward
    out.requires_grad = requires_grad
    return out
