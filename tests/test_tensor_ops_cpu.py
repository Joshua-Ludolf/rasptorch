import numpy as np

from rasptorch import Tensor


def test_basic_ops_cpu_match_numpy() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 3), dtype=np.float32)
    b = rng.standard_normal((4, 3), dtype=np.float32)

    ta = Tensor(a)
    tb = Tensor(b)

    np.testing.assert_allclose((ta + tb).numpy(), a + b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose((ta * tb).numpy(), a * b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose((ta - tb).numpy(), a - b, rtol=1e-6, atol=1e-6)


def test_matmul_cpu_match_numpy() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((5, 7), dtype=np.float32)
    b = rng.standard_normal((7, 2), dtype=np.float32)

    ta = Tensor(a)
    tb = Tensor(b)

    np.testing.assert_allclose((ta @ tb).numpy(), a @ b, rtol=1e-5, atol=1e-6)


def test_relu_sum_mean_cpu() -> None:
    x_np = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    x = Tensor(x_np)
    np.testing.assert_allclose(x.relu().numpy(), np.maximum(x_np, 0.0))
    np.testing.assert_allclose(float(x.sum().numpy().reshape(-1)[0]), float(x_np.sum()), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(float(x.mean().numpy().reshape(-1)[0]), float(x_np.mean()), rtol=1e-6, atol=1e-6)


def test_rowvec_broadcast_cpu_forward() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((6, 4), dtype=np.float32)
    b = rng.standard_normal((4,), dtype=np.float32)

    ta = Tensor(a)
    tb = Tensor(b)

    out_add = ta.add_rowvec(tb)
    out_mul = ta.mul_rowvec(tb)

    np.testing.assert_allclose(out_add.numpy(), a + b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_mul.numpy(), a * b, rtol=1e-6, atol=1e-6)
