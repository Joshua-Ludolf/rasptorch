import numpy as np
import pytest

import rasptorch
from rasptorch import Tensor
from rasptorch import functional as F
from rasptorch import vulkan_backend as vk
from rasptorch.nn import LayerNorm


def _grad_to_numpy(t: Tensor) -> np.ndarray:
    if t.grad is not None:
        return np.asarray(t.grad, dtype=np.float32)
    if t.grad_vkbuf is not None:
        return np.asarray(vk.to_cpu(t.grad_vkbuf), dtype=np.float32)
    raise AssertionError("expected tensor to have a grad")


def test_gpu_log_softmax_backward_matches_cpu() -> None:
    rng = np.random.default_rng(0)
    N, C = 8, 5
    logits_np = rng.standard_normal((N, C), dtype=np.float32)
    labels = rng.integers(0, C, size=(N,), dtype=np.int64)

    target_cpu = F.one_hot(labels, C)

    # CPU reference
    logits_cpu = Tensor(logits_np.copy(), requires_grad=True)
    logp_cpu = F.log_softmax(logits_cpu, dim=1)
    loss_cpu = -(target_cpu * logp_cpu).sum() / float(N)
    loss_cpu.backward()

    # GPU path (Vulkan if available, otherwise NumPy fallback)
    logits_gpu = Tensor(logits_np.copy(), requires_grad=True).to("gpu")
    target_gpu = target_cpu.to("gpu")
    logp_gpu = F.log_softmax(logits_gpu, dim=1)
    loss_gpu = -(target_gpu * logp_gpu).sum() / float(N)
    loss_gpu.backward()

    np.testing.assert_allclose(loss_gpu.numpy(), loss_cpu.numpy(), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(_grad_to_numpy(logits_gpu), logits_cpu.grad, rtol=2e-2, atol=2e-2)


def test_gpu_softmax_forward_matches_cpu() -> None:
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((6, 7), dtype=np.float32)

    x_cpu = Tensor(x_np.copy(), requires_grad=False)
    y_cpu = F.softmax(x_cpu, dim=1).numpy()

    x_gpu = Tensor(x_np.copy(), requires_grad=False).to("gpu")
    y_gpu = F.softmax(x_gpu, dim=1).numpy()

    np.testing.assert_allclose(y_gpu, y_cpu, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(y_gpu.sum(axis=1), np.ones((x_np.shape[0],), dtype=np.float32), rtol=2e-3, atol=2e-3)


def test_gpu_layernorm_affine_grads_match_cpu() -> None:
    rng = np.random.default_rng(2)
    N, H = 10, 12

    x_np = rng.standard_normal((N, H), dtype=np.float32)

    # Build CPU and GPU modules with identical params.
    ln_cpu = LayerNorm(H, eps=1e-5, elementwise_affine=True)
    ln_gpu = LayerNorm(H, eps=1e-5, elementwise_affine=True)

    w = rng.standard_normal((H,), dtype=np.float32)
    b = rng.standard_normal((H,), dtype=np.float32)
    assert ln_cpu.weight is not None and ln_cpu.bias is not None
    assert ln_gpu.weight is not None and ln_gpu.bias is not None
    ln_cpu.weight.data = w.copy()
    ln_cpu.bias.data = b.copy()
    ln_gpu.weight.data = w.copy()
    ln_gpu.bias.data = b.copy()

    x_cpu = Tensor(x_np.copy(), requires_grad=True)
    y_cpu = ln_cpu(x_cpu)
    loss_cpu = y_cpu.mean()
    loss_cpu.backward()

    ln_gpu = ln_gpu.to("gpu")
    x_gpu = Tensor(x_np.copy(), requires_grad=True).to("gpu")
    y_gpu = ln_gpu(x_gpu)
    loss_gpu = y_gpu.mean()
    loss_gpu.backward()

    np.testing.assert_allclose(loss_gpu.numpy(), loss_cpu.numpy(), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(_grad_to_numpy(x_gpu), x_cpu.grad, rtol=5e-2, atol=5e-2)

    # Param grads (affine)
    assert ln_gpu.weight is not None and ln_gpu.bias is not None
    gw = _grad_to_numpy(ln_gpu.weight)
    gb = _grad_to_numpy(ln_gpu.bias)
    np.testing.assert_allclose(gw, ln_cpu.weight.grad, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(gb, ln_cpu.bias.grad, rtol=5e-2, atol=5e-2)


def test_no_grad_and_detach_on_gpu_block_grads() -> None:
    rng = np.random.default_rng(3)

    x = Tensor(rng.standard_normal((4, 4), dtype=np.float32), requires_grad=True).to("gpu")

    # detach should stop gradients
    y = (x * 2.0).detach()
    z = (y * 3.0).sum()
    z.backward()
    assert x.grad is None
    assert x.grad_vkbuf is None

    # no_grad should avoid tracking entirely
    w = Tensor(rng.standard_normal((4, 4), dtype=np.float32), requires_grad=True).to("gpu")
    with rasptorch.no_grad():
        loss = (w * w).mean()

    assert loss.requires_grad is False
    loss.backward()
    assert w.grad is None
    assert w.grad_vkbuf is None
