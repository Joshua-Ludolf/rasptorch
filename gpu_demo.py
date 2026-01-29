import argparse
import numpy as np
import time

from rasptorch import Tensor
from rasptorch import vulkan_backend as vk


def _stats_ms(samples_ns: list[int]) -> dict[str, float]:
    arr = (np.array(samples_ns, dtype=np.float64) / 1e6)  # ms
    return {
        "n": float(arr.size),
        "min_ms": float(arr.min()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
    }


def _fmt_stats(label: str, s: dict[str, float]) -> str:
    return (
        f"{label}: n={int(s['n'])} "
        f"min={s['min_ms']:.3f}ms "
        f"p50={s['p50_ms']:.3f}ms "
        f"p95={s['p95_ms']:.3f}ms "
        f"mean={s['mean_ms']:.3f}ms ±{s['std_ms']:.3f}ms"
    )


def _bench(fn, warmup: int, iters: int) -> list[int]:
    for _ in range(warmup):
        fn()
    times: list[int] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    return times


def _assert_close(name: str, got: np.ndarray, expected: np.ndarray, *, rtol: float = 1e-4, atol: float = 1e-5) -> None:
    got = np.asarray(got)
    expected = np.asarray(expected)
    if got.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch {got.shape} vs {expected.shape}")
    if not np.allclose(got, expected, rtol=rtol, atol=atol):
        max_abs = float(np.max(np.abs(got - expected)))
        raise AssertionError(f"{name}: values differ (max_abs={max_abs})")


def run_smoke_tests() -> None:
    """Fast correctness smoke tests for the Vulkan kernels used by demos/training."""

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
            _assert_close("elemwise", vk.to_cpu(out), np.maximum(x * y + x, 0.0))
        finally:
            vk.free(tmp)
            vk.free(tmp2)
            vk.free(out)
    finally:
        vk.free(a)
        vk.free(b)

    # Matmul
    a_np = rng.standard_normal((17, 19), dtype=np.float32)
    b_np = rng.standard_normal((19, 23), dtype=np.float32)
    a = vk.to_gpu(a_np)
    b = vk.to_gpu(b_np)
    try:
        out = vk.matmul(a, b)
        try:
            _assert_close("matmul", vk.to_cpu(out), a_np @ b_np, rtol=2e-3, atol=1e-3)
        finally:
            vk.free(out)
    finally:
        vk.free(a)
        vk.free(b)

    # Transpose2d
    t_np = rng.standard_normal((7, 5), dtype=np.float32)
    t = vk.to_gpu(t_np)
    try:
        out = vk.transpose2d(t)
        try:
            _assert_close("transpose2d", vk.to_cpu(out), t_np.T)
        finally:
            vk.free(out)
    finally:
        vk.free(t)

    # Training kernels: add_rowvec + reduce_sum_rows + relu_backward + mse_grad
    m = rng.standard_normal((11, 13), dtype=np.float32)
    rv = rng.standard_normal((13,), dtype=np.float32)
    m_buf = vk.to_gpu(m)
    rv_buf = vk.to_gpu(rv)
    try:
        out = vk.add_rowvec(m_buf, rv_buf)
        try:
            _assert_close("add_rowvec", vk.to_cpu(out), m + rv)
        finally:
            vk.free(out)

        rs = vk.reduce_sum_rows(m_buf)
        try:
            _assert_close("reduce_sum_rows", vk.to_cpu(rs), m.sum(axis=0))
        finally:
            vk.free(rs)

        grad_out_np = rng.standard_normal(m.shape, dtype=np.float32)
        grad_out = vk.to_gpu(grad_out_np)
        try:
            gi = vk.relu_backward(grad_out, m_buf)
            try:
                _assert_close("relu_backward", vk.to_cpu(gi), grad_out_np * (m > 0))
            finally:
                vk.free(gi)
        finally:
            vk.free(grad_out)

        pred = rng.standard_normal((9, 4), dtype=np.float32)
        target = rng.standard_normal((9, 4), dtype=np.float32)
        pred_b = vk.to_gpu(pred)
        tgt_b = vk.to_gpu(target)
        try:
            mg = vk.mse_grad(pred_b, tgt_b)
            try:
                _assert_close("mse_grad", vk.to_cpu(mg), 2.0 * (pred - target) / pred.size)
            finally:
                vk.free(mg)
        finally:
            vk.free(pred_b)
            vk.free(tgt_b)

    finally:
        vk.free(m_buf)
        vk.free(rv_buf)


def run_benchmarks() -> None:
    # Keep these modest so it runs quickly on a Pi.
    warmup = 5
    iters = 50

    rng = np.random.default_rng(0)

    # -----------------
    # Elementwise + ReLU
    # -----------------
    n = 1_000_000
    x_np = np.linspace(-1, 1, n, dtype=np.float32).reshape(-1, 1)
    y_np = (np.ones((n, 1), dtype=np.float32) * 3.0)

    def cpu_elemwise() -> np.ndarray:
        return np.maximum(x_np * y_np + x_np, 0.0)

    t0 = time.perf_counter_ns()
    x_gpu = Tensor(x_np).to("gpu")
    y_gpu = Tensor(y_np).to("gpu")
    t1 = time.perf_counter_ns()
    upload_ms = (t1 - t0) / 1e6

    a_buf = x_gpu._as_vkbuf()
    b_buf = y_gpu._as_vkbuf()

    def gpu_elemwise_with_readback() -> np.ndarray:
        # Full path: GPU compute + CPU readback per iteration.
        tmp = vk.mul(a_buf, b_buf)
        tmp2 = vk.add(tmp, a_buf)
        out = vk.relu(tmp2)
        try:
            return vk.to_cpu(out)
        finally:
            vk.free(tmp)
            vk.free(tmp2)
            vk.free(out)

    # Compute-only: allocate/free each iteration to avoid leaks (includes alloc overhead).
    out_hold: vk.VulkanBuffer | None = None

    def gpu_elemwise_compute_only() -> None:
        nonlocal out_hold
        if out_hold is not None:
            vk.free(out_hold)
        tmp = vk.mul(a_buf, b_buf)
        tmp2 = vk.add(tmp, a_buf)
        out_hold = vk.relu(tmp2)
        vk.free(tmp)
        vk.free(tmp2)

    def gpu_elemwise_fused_with_readback() -> np.ndarray:
        out = vk.mul_add_relu(a_buf, b_buf)
        try:
            return vk.to_cpu(out)
        finally:
            vk.free(out)

    # Fused backend with preallocated output (no per-iter allocations).
    out_buf = vk.empty(a_buf.shape)

    def gpu_elemwise_fused_compute_only_no_alloc() -> None:
        vk.mul_add_relu_out(a_buf, b_buf, out_buf)

    cpu_times = _bench(cpu_elemwise, warmup, iters)
    gpu_rw_times = _bench(gpu_elemwise_with_readback, warmup, iters)
    gpu_co_times = _bench(gpu_elemwise_compute_only, warmup, iters)
    gpu_fused_rw_times = _bench(gpu_elemwise_fused_with_readback, warmup, iters)
    gpu_fused_no_alloc_times = _bench(gpu_elemwise_fused_compute_only_no_alloc, warmup, iters)

    # One-time download timing (after compute-only benchmark)
    t0 = time.perf_counter_ns()
    _ = (vk.to_cpu(out_hold) if out_hold is not None else None)
    t1 = time.perf_counter_ns()
    download_ms = (t1 - t0) / 1e6

    cpu_s = _stats_ms(cpu_times)
    gpu_rw_s = _stats_ms(gpu_rw_times)
    gpu_co_s = _stats_ms(gpu_co_times)
    gpu_fused_rw_s = _stats_ms(gpu_fused_rw_times)
    gpu_fused_no_alloc_s = _stats_ms(gpu_fused_no_alloc_times)
    speedup_rw = cpu_s["mean_ms"] / gpu_rw_s["mean_ms"] if gpu_rw_s["mean_ms"] > 0 else float("inf")
    speedup_co = cpu_s["mean_ms"] / gpu_co_s["mean_ms"] if gpu_co_s["mean_ms"] > 0 else float("inf")
    speedup_fused_rw = (
        cpu_s["mean_ms"] / gpu_fused_rw_s["mean_ms"] if gpu_fused_rw_s["mean_ms"] > 0 else float("inf")
    )
    speedup_fused_no_alloc = (
        cpu_s["mean_ms"] / gpu_fused_no_alloc_s["mean_ms"]
        if gpu_fused_no_alloc_s["mean_ms"] > 0
        else float("inf")
    )

    print("\n=== Benchmark: elementwise (x*y + x) + ReLU ===")
    print(f"shape: {x_np.shape}  dtype: float32")
    print(f"GPU upload once: {upload_ms:.3f}ms")
    print(f"GPU download once: {download_ms:.3f}ms")
    print(_fmt_stats("CPU (NumPy)", cpu_s))
    print(_fmt_stats("GPU (Vulkan backend) compute+readback", gpu_rw_s))
    print(f"speedup (mean): {speedup_rw:.2f}x")
    print(_fmt_stats("GPU (Vulkan backend) compute-only", gpu_co_s))
    print(f"speedup (mean, compute-only): {speedup_co:.2f}x")
    print(_fmt_stats("GPU (Vulkan backend) fused compute+readback", gpu_fused_rw_s))
    print(f"speedup (mean, fused): {speedup_fused_rw:.2f}x")
    print(_fmt_stats("GPU (Vulkan backend) fused compute-only (no alloc)", gpu_fused_no_alloc_s))
    print(f"speedup (mean, fused no-alloc): {speedup_fused_no_alloc:.2f}x")

    # ------
    # Matmul
    # ------
    m, k, p = 256, 256, 256
    a_np = rng.standard_normal((m, k), dtype=np.float32)
    b_np = rng.standard_normal((k, p), dtype=np.float32)

    def cpu_matmul() -> np.ndarray:
        return a_np @ b_np

    t0 = time.perf_counter_ns()
    a_gpu = Tensor(a_np).to("gpu")
    b_gpu = Tensor(b_np).to("gpu")
    t1 = time.perf_counter_ns()
    upload_ms = (t1 - t0) / 1e6

    a_buf = a_gpu._as_vkbuf()
    b_buf = b_gpu._as_vkbuf()

    def gpu_matmul_with_readback() -> np.ndarray:
        out = vk.matmul(a_buf, b_buf)
        try:
            return vk.to_cpu(out)
        finally:
            vk.free(out)

    out_hold: vk.VulkanBuffer | None = None

    def gpu_matmul_compute_only() -> None:
        nonlocal out_hold
        if out_hold is not None:
            vk.free(out_hold)
        out_hold = vk.matmul(a_buf, b_buf)

    # Matmul with preallocated output (no per-iter allocations).
    c_buf = vk.empty((m, p))

    def gpu_matmul_compute_only_no_alloc() -> None:
        vk.matmul_out(a_buf, b_buf, c_buf)

    cpu_times = _bench(cpu_matmul, warmup, iters)
    gpu_rw_times = _bench(gpu_matmul_with_readback, warmup, iters)
    gpu_co_times = _bench(gpu_matmul_compute_only, warmup, iters)
    gpu_no_alloc_times = _bench(gpu_matmul_compute_only_no_alloc, warmup, iters)

    t0 = time.perf_counter_ns()
    _ = (vk.to_cpu(out_hold) if out_hold is not None else None)
    t1 = time.perf_counter_ns()
    download_ms = (t1 - t0) / 1e6

    cpu_s = _stats_ms(cpu_times)
    gpu_rw_s = _stats_ms(gpu_rw_times)
    gpu_co_s = _stats_ms(gpu_co_times)
    gpu_no_alloc_s = _stats_ms(gpu_no_alloc_times)
    speedup_rw = cpu_s["mean_ms"] / gpu_rw_s["mean_ms"] if gpu_rw_s["mean_ms"] > 0 else float("inf")
    speedup_co = cpu_s["mean_ms"] / gpu_co_s["mean_ms"] if gpu_co_s["mean_ms"] > 0 else float("inf")
    speedup_no_alloc = (
        cpu_s["mean_ms"] / gpu_no_alloc_s["mean_ms"] if gpu_no_alloc_s["mean_ms"] > 0 else float("inf")
    )

    print("\n=== Benchmark: matmul (A @ B) ===")
    print(f"A: {a_np.shape}  B: {b_np.shape}  dtype: float32")
    print(f"GPU upload once: {upload_ms:.3f}ms")
    print(f"GPU download once: {download_ms:.3f}ms")
    print(_fmt_stats("CPU (NumPy)", cpu_s))
    print(_fmt_stats("GPU (Vulkan backend) compute+readback", gpu_rw_s))
    print(f"speedup (mean): {speedup_rw:.2f}x")
    print(_fmt_stats("GPU (Vulkan backend) compute-only", gpu_co_s))
    print(f"speedup (mean, compute-only): {speedup_co:.2f}x")
    print(_fmt_stats("GPU (Vulkan backend) compute-only (no alloc)", gpu_no_alloc_s))
    print(f"speedup (mean, no-alloc): {speedup_no_alloc:.2f}x")


def run_gpu_demo() -> None:
    # Simple elementwise ops on the "gpu" device.
    x_cpu = Tensor(np.linspace(-1, 1, 8, dtype="float32").reshape(-1, 1))
    y_cpu = Tensor(np.ones((8, 1), dtype="float32") * 3)

    x = x_cpu.to("gpu")
    y = y_cpu.to("gpu")

    z = (x * y + x).relu()

    print("x (cpu):", x_cpu.numpy().ravel())
    print("y (cpu):", y_cpu.numpy().ravel())
    print("z (gpu -> cpu):", z.numpy().ravel())

    # Simple matrix multiply on "gpu".
    a_cpu = Tensor(np.random.randn(4, 3).astype("float32"))
    b_cpu = Tensor(np.random.randn(3, 2).astype("float32"))

    a = a_cpu.to("gpu")
    b = b_cpu.to("gpu")

    c = a @ b
    print("A @ B (gpu -> cpu) shape:", c.shape)
    print("A @ B (gpu -> cpu):")
    print(c.numpy())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="rasptorch GPU demo/bench")
    p.add_argument("--smoke-only", action="store_true", help="Run correctness smoke tests only")
    p.add_argument("--bench-only", action="store_true", help="Run benchmarks only")
    args = p.parse_args()

    try:
        vk.init(strict=True)
    except RuntimeError as e:
        print("Vulkan init failed:")
        print(f"  {e}")
        print(
            "Tip: ensure Vulkan is installed and working, the Python 'vulkan' package is available, "
            "and 'glslc' (shader compiler) is installed on the system."
        )
        raise SystemExit(1) from e

    if not args.bench_only:
        print("=== Smoke tests ===")
        run_smoke_tests()
        print("PASS")

    if not args.smoke_only:
        run_gpu_demo()
        run_benchmarks()
