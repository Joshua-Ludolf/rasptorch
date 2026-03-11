import numpy as np

from rasptorch import Tensor, cat, stack


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


def test_shape_manipulation_ops_match_numpy() -> None:
    x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = Tensor(x_np)

    np.testing.assert_allclose(x.unsqueeze(1).numpy(), np.expand_dims(x_np, axis=1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x.unsqueeze(-1).numpy(), np.expand_dims(x_np, axis=-1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x.unsqueeze(1).squeeze(1).numpy(), x_np, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x.permute(2, 0, 1).numpy(), np.transpose(x_np, (2, 0, 1)), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x.transpose(0, 2).numpy(), np.swapaxes(x_np, 0, 2), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x.flatten(1).numpy(), x_np.reshape(2, 12), rtol=1e-6, atol=1e-6)


def test_cat_stack_split_chunk_match_numpy() -> None:
    a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    b_np = (np.arange(6, dtype=np.float32) + 10).reshape(2, 3)
    a = Tensor(a_np)
    b = Tensor(b_np)

    np.testing.assert_allclose(cat([a, b], dim=0).numpy(), np.concatenate([a_np, b_np], axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cat([a, b], dim=1).numpy(), np.concatenate([a_np, b_np], axis=1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(stack([a, b], dim=0).numpy(), np.stack([a_np, b_np], axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(stack([a, b], dim=1).numpy(), np.stack([a_np, b_np], axis=1), rtol=1e-6, atol=1e-6)

    splits = a.split(2, dim=1)
    assert len(splits) == 2
    np.testing.assert_allclose(splits[0].numpy(), a_np[:, :2], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(splits[1].numpy(), a_np[:, 2:], rtol=1e-6, atol=1e-6)

    chunks = Tensor(np.arange(10, dtype=np.float32)).chunk(3, dim=0)
    assert len(chunks) == 3
    np.testing.assert_allclose(chunks[0].numpy(), np.array([0, 1, 2, 3], dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(chunks[1].numpy(), np.array([4, 5, 6, 7], dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(chunks[2].numpy(), np.array([8, 9], dtype=np.float32), rtol=1e-6, atol=1e-6)
