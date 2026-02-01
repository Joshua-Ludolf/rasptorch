import numpy as np

from rasptorch import Tensor
from rasptorch import functional as F
from rasptorch.nn import Linear


def _finite_diff_grad(f, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = float(x[idx])
        x[idx] = np.float32(orig + eps)
        fp = float(f(x))
        x[idx] = np.float32(orig - eps)
        fm = float(f(x))
        x[idx] = np.float32(orig)
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def test_grad_matmul_sum() -> None:
    rng = np.random.default_rng(0)
    a0 = rng.standard_normal((2, 3), dtype=np.float32)
    b0 = rng.standard_normal((3, 4), dtype=np.float32)

    a = Tensor(a0, requires_grad=True)
    b = Tensor(b0, requires_grad=True)
    y = (a @ b).sum()
    y.backward()

    def f_a(x):
        return float((x @ b0).sum())

    def f_b(x):
        return float((a0 @ x).sum())

    np.testing.assert_allclose(a.grad, _finite_diff_grad(f_a, a0), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(b.grad, _finite_diff_grad(f_b, b0), rtol=2e-2, atol=2e-2)


def test_grad_relu_sum() -> None:
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((3, 3), dtype=np.float32)

    x = Tensor(x0, requires_grad=True)
    y = x.relu().sum()
    y.backward()

    def f_x(z):
        return float(np.maximum(z, 0.0).sum())

    np.testing.assert_allclose(x.grad, _finite_diff_grad(f_x, x0), rtol=1e-2, atol=1e-2)


def test_grad_linear_sum() -> None:
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((4, 5), dtype=np.float32)

    layer = Linear(5, 3)
    layer.weight.requires_grad = False
    if layer.bias is not None:
        layer.bias.requires_grad = False

    x = Tensor(x0, requires_grad=True)
    y = layer(x).sum()
    y.backward()

    w = layer.weight.data

    def f_x(z):
        return float((z @ w.T).sum())

    np.testing.assert_allclose(x.grad, _finite_diff_grad(f_x, x0), rtol=2e-2, atol=2e-2)


def test_grad_mse_loss() -> None:
    rng = np.random.default_rng(3)
    pred0 = rng.standard_normal((3, 2), dtype=np.float32)
    tgt0 = rng.standard_normal((3, 2), dtype=np.float32)

    pred = Tensor(pred0, requires_grad=True)
    tgt = Tensor(tgt0)
    loss = F.mse_loss(pred, tgt)
    loss.backward()

    def f_pred(z):
        return float(((z - tgt0) ** 2).mean())

    np.testing.assert_allclose(pred.grad, _finite_diff_grad(f_pred, pred0), rtol=2e-2, atol=2e-2)


def test_grad_cross_entropy_logits() -> None:
    rng = np.random.default_rng(4)
    N, C = 4, 5
    logits0 = rng.standard_normal((N, C), dtype=np.float32)
    y = rng.integers(0, C, size=(N,), dtype=np.int64)
    target = F.one_hot(y, C)

    logits = Tensor(logits0, requires_grad=True)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    def f_logits(z):
        m = z.max(axis=1, keepdims=True)
        zz = z - m
        lse = np.log(np.exp(zz).sum(axis=1, keepdims=True)) + m
        loss_vec = -(target.numpy() * (z - lse)).sum(axis=1)
        return float(loss_vec.mean())

    np.testing.assert_allclose(logits.grad, _finite_diff_grad(f_logits, logits0), rtol=3e-2, atol=3e-2)
