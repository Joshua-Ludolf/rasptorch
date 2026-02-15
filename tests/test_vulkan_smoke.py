import numpy as np
import pytest

from rasptorch import vulkan_backend as vk


def test_vulkan_backend_smoke() -> None:
    """Backend smoke test.

    This test always runs:
    - If Vulkan is available, it uses strict init and exercises real GPU kernels.
    - If Vulkan is unavailable, it exercises the NumPy fallback path via the same API.
    """

    try:
        vk.init(strict=True)
    except Exception:
        # Don't skip on non-Vulkan environments; the backend has a NumPy fallback.
        vk.init(strict=False)

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

    # NN essentials: softmax/log_softmax (+ backward) and layernorm2d
    logits = rng.standard_normal((9, 7), dtype=np.float32)
    go_np = rng.standard_normal((9, 7), dtype=np.float32)
    lg = vk.to_gpu(logits)
    go = vk.to_gpu(go_np)
    try:
        y = vk.softmax2d(lg)
        try:
            got = vk.to_cpu(y)
        finally:
            vk.free(y)

        z = logits - logits.max(axis=1, keepdims=True)
        y_np = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        assert np.allclose(got, y_np, rtol=2e-3, atol=2e-3)

        # softmax backward
        yb = vk.to_gpu(y_np.astype(np.float32))
        try:
            dx = vk.softmax2d_backward(yb, go)
            try:
                got = vk.to_cpu(dx)
            finally:
                vk.free(dx)
        finally:
            vk.free(yb)

        dot = (go_np * y_np).sum(axis=1, keepdims=True)
        dx_np = y_np * (go_np - dot)
        assert np.allclose(got, dx_np, rtol=2e-3, atol=2e-3)

        # log_softmax forward
        lp = vk.log_softmax2d(lg)
        try:
            got = vk.to_cpu(lp)
        finally:
            vk.free(lp)

        m = logits.max(axis=1, keepdims=True)
        zz = logits - m
        lse = np.log(np.exp(zz).sum(axis=1, keepdims=True)) + m
        lp_np = logits - lse
        assert np.allclose(got, lp_np, rtol=2e-3, atol=2e-3)

        # log_softmax backward
        lpb = vk.to_gpu(lp_np.astype(np.float32))
        try:
            dx = vk.log_softmax2d_backward(lpb, go)
            try:
                got = vk.to_cpu(dx)
            finally:
                vk.free(dx)
        finally:
            vk.free(lpb)

        s = np.exp(lp_np)
        sumg = go_np.sum(axis=1, keepdims=True)
        dx_np = go_np - s * sumg
        assert np.allclose(got, dx_np, rtol=2e-3, atol=2e-3)

        # layernorm2d forward
        ln = vk.layernorm2d(lg)
        try:
            got = vk.to_cpu(ln)
        finally:
            vk.free(ln)

        mean = got.mean(axis=1)
        var = got.var(axis=1)
        assert np.allclose(mean, 0.0, atol=1e-2)
        assert np.allclose(var, 1.0, atol=5e-2)

        # layernorm2d backward (grad wrt normalized output)
        dx = vk.layernorm2d_backward(lg, go)
        try:
            got = vk.to_cpu(dx)
        finally:
            vk.free(dx)

        mean = logits.mean(axis=1, keepdims=True)
        var = ((logits - mean) ** 2).mean(axis=1, keepdims=True)
        invstd = 1.0 / np.sqrt(var + 1e-5)
        xhat = (logits - mean) * invstd
        sum1 = go_np.sum(axis=1, keepdims=True)
        sum2 = (go_np * xhat).sum(axis=1, keepdims=True)
        C = logits.shape[1]
        dx_np = (invstd / max(1, C)) * (go_np * C - sum1 - xhat * sum2)
        assert np.allclose(got, dx_np, rtol=2e-3, atol=2e-3)

    finally:
        vk.free(lg)
        vk.free(go)
