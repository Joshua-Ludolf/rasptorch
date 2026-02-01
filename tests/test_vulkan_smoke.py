import numpy as np
import pytest

from rasptorch import vulkan_backend as vk


def test_vulkan_smoke_if_available() -> None:
    # Skip gracefully on non-Pi/CI where Vulkan bindings or drivers are absent.
    try:
        vk.init(strict=True)
    except Exception:
        pytest.skip(vk.disabled_reason() or "Vulkan unavailable")

    rng = np.random.default_rng(0)

    # Elementwise: (x*y + x).relu()
    x = rng.standard_normal((33, 17), dtype=np.float32)
    y = rng.standard_normal((33, 17), dtype=np.float32)
    a = vk.to_gpu(x)
    b = vk.to_gpu(y)
    try:
        tmp = vk.mul(a, b)
        tmp2 = vk.add(tmp, a)
        out = vk.relu(tmp2)
        try:
            got = vk.to_cpu(out)
        finally:
            vk.free(tmp)
            vk.free(tmp2)
            vk.free(out)
    finally:
        vk.free(a)
        vk.free(b)

    assert np.allclose(got, np.maximum(x * y + x, 0.0), rtol=2e-3, atol=1e-3)

    # Matmul
    a_np = rng.standard_normal((17, 19), dtype=np.float32)
    b_np = rng.standard_normal((19, 23), dtype=np.float32)
    aa = vk.to_gpu(a_np)
    bb = vk.to_gpu(b_np)
    try:
        out = vk.matmul(aa, bb)
        try:
            got = vk.to_cpu(out)
        finally:
            vk.free(out)
    finally:
        vk.free(aa)
        vk.free(bb)

    assert got.shape == (17, 23)
    assert np.allclose(got, a_np @ b_np, rtol=2e-3, atol=1e-3)

    # Transpose2d
    t_np = rng.standard_normal((7, 5), dtype=np.float32)
    t = vk.to_gpu(t_np)
    try:
        out = vk.transpose2d(t)
        try:
            got = vk.to_cpu(out)
        finally:
            vk.free(out)
    finally:
        vk.free(t)

    assert np.array_equal(got, t_np.T)

    # Training kernels: add_rowvec + reduce_sum_rows + relu_backward + mse_grad
    m = rng.standard_normal((11, 13), dtype=np.float32)
    rv = rng.standard_normal((13,), dtype=np.float32)
    m_buf = vk.to_gpu(m)
    rv_buf = vk.to_gpu(rv)
    try:
        out = vk.add_rowvec(m_buf, rv_buf)
        try:
            got = vk.to_cpu(out)
        finally:
            vk.free(out)
        assert np.allclose(got, m + rv, rtol=2e-3, atol=1e-3)

        rs = vk.reduce_sum_rows(m_buf)
        try:
            got = vk.to_cpu(rs)
        finally:
            vk.free(rs)
        assert np.allclose(got, m.sum(axis=0), rtol=2e-3, atol=1e-3)

        grad_out_np = rng.standard_normal(m.shape, dtype=np.float32)
        grad_out = vk.to_gpu(grad_out_np)
        try:
            gi = vk.relu_backward(grad_out, m_buf)
            try:
                got = vk.to_cpu(gi)
            finally:
                vk.free(gi)
        finally:
            vk.free(grad_out)
        assert np.allclose(got, grad_out_np * (m > 0), rtol=2e-3, atol=1e-3)

        pred = rng.standard_normal((9, 4), dtype=np.float32)
        target = rng.standard_normal((9, 4), dtype=np.float32)
        pred_b = vk.to_gpu(pred)
        tgt_b = vk.to_gpu(target)
        try:
            mg = vk.mse_grad(pred_b, tgt_b)
            try:
                got = vk.to_cpu(mg)
            finally:
                vk.free(mg)
        finally:
            vk.free(pred_b)
            vk.free(tgt_b)
        assert np.allclose(got, 2.0 * (pred - target) / pred.size, rtol=2e-3, atol=1e-3)

    finally:
        vk.free(m_buf)
        vk.free(rv_buf)
