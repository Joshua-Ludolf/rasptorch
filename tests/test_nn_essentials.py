import numpy as np

from rasptorch import Tensor, no_grad
from rasptorch import functional as F
from rasptorch.nn import Dropout, LayerNorm


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


def test_softmax_forward_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((4, 7), dtype=np.float32)
    x = Tensor(x0)
    y = F.softmax(x, dim=1).numpy()

    z = x0 - x0.max(axis=1, keepdims=True)
    y_np = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(y, y_np, rtol=1e-6, atol=1e-6)


def test_log_softmax_forward_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((3, 5), dtype=np.float32)
    x = Tensor(x0)
    y = F.log_softmax(x, dim=-1).numpy()

    m = x0.max(axis=1, keepdims=True)
    z = x0 - m
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True)) + m
    y_np = x0 - lse
    np.testing.assert_allclose(y, y_np, rtol=1e-6, atol=1e-6)


def test_softmax_grad_matches_finite_diff() -> None:
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((2, 4), dtype=np.float32)
    w0 = rng.standard_normal((2, 4), dtype=np.float32)

    x = Tensor(x0, requires_grad=True)
    w = Tensor(w0)
    y = (F.softmax(x, dim=1) * w).sum()
    y.backward()

    def f_x(z):
        z = np.asarray(z, dtype=np.float32)
        zz = z - z.max(axis=1, keepdims=True)
        s = np.exp(zz) / np.exp(zz).sum(axis=1, keepdims=True)
        return float((s * w0).sum())

    np.testing.assert_allclose(x.grad, _finite_diff_grad(f_x, x0), rtol=3e-2, atol=3e-2)


def test_log_softmax_grad_matches_finite_diff() -> None:
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal((2, 5), dtype=np.float32)
    w0 = rng.standard_normal((2, 5), dtype=np.float32)

    x = Tensor(x0, requires_grad=True)
    w = Tensor(w0)
    y = (F.log_softmax(x, dim=1) * w).sum()
    y.backward()

    def f_x(z):
        z = np.asarray(z, dtype=np.float32)
        m = z.max(axis=1, keepdims=True)
        zz = z - m
        lse = np.log(np.exp(zz).sum(axis=1, keepdims=True)) + m
        ls = z - lse
        return float((ls * w0).sum())

    np.testing.assert_allclose(x.grad, _finite_diff_grad(f_x, x0), rtol=3e-2, atol=3e-2)


def test_dropout_train_vs_eval() -> None:
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal((128, 16), dtype=np.float32)
    x = Tensor(x0)

    d = Dropout(p=0.5)
    d.train(True)
    y = d(x).numpy()
    # Expect many zeros and preserved mean (in expectation).
    assert (y == 0.0).mean() > 0.3
    np.testing.assert_allclose(y.mean(), x0.mean(), rtol=0.25, atol=0.25)

    d.eval()
    y2 = d(x).numpy()
    np.testing.assert_allclose(y2, x0, rtol=0.0, atol=0.0)


def test_layernorm_forward_mean_var() -> None:
    rng = np.random.default_rng(5)
    x0 = rng.standard_normal((8, 32), dtype=np.float32)
    x = Tensor(x0)

    ln = LayerNorm(32)
    ln.eval()
    y = ln(x).numpy()
    m = y.mean(axis=1)
    v = y.var(axis=1)
    np.testing.assert_allclose(m, np.zeros_like(m), atol=3e-3, rtol=0)
    np.testing.assert_allclose(v, np.ones_like(v), atol=5e-2, rtol=0)


def test_layernorm_grad_matches_finite_diff() -> None:
    rng = np.random.default_rng(6)
    x0 = rng.standard_normal((3, 6), dtype=np.float32)
    w0 = rng.standard_normal((6,), dtype=np.float32)

    x = Tensor(x0, requires_grad=True)
    ln = LayerNorm(6)
    ln.weight.data[...] = w0
    ln.bias.data[...] = 0.0
    ln.weight.grad = None
    ln.bias.grad = None

    y = (ln(x).sum())
    y.backward()

    def f_x(z):
        z = np.asarray(z, dtype=np.float32)
        mean = z.mean(axis=1, keepdims=True)
        var = ((z - mean) ** 2).mean(axis=1, keepdims=True)
        invstd = 1.0 / np.sqrt(var + 1e-5)
        xhat = (z - mean) * invstd
        out = xhat * w0.reshape(1, -1)
        return float(out.sum())

    np.testing.assert_allclose(x.grad, _finite_diff_grad(f_x, x0), rtol=4e-2, atol=4e-2)


def test_softmax_no_grad_disables_tracking() -> None:
    rng = np.random.default_rng(7)
    x = Tensor(rng.standard_normal((2, 3), dtype=np.float32), requires_grad=True)
    with no_grad():
        y = F.softmax(x)
        assert y.requires_grad is False
