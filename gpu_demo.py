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

    def gpu_elemwise_with_readback() -> np.ndarray:
        # Full path: GPU compute + CPU readback per iteration.
        return (x_gpu * y_gpu + x_gpu).relu().numpy()

    # Compute-only: no per-iter .numpy() call (download once after the benchmark).
    z_hold: Tensor | None = None

    def gpu_elemwise_compute_only() -> None:
        nonlocal z_hold
        z_hold = (x_gpu * y_gpu + x_gpu).relu()

    # Fused backend: single dispatch for (x*y + x) then ReLU.
    fused_hold: Tensor | None = None

    def gpu_elemwise_fused_compute_only() -> None:
        nonlocal fused_hold
        fused_hold = Tensor._from_vkbuf(vk.mul_add_relu(x_gpu._as_vkbuf(), y_gpu._as_vkbuf()))

    def gpu_elemwise_fused_with_readback() -> np.ndarray:
        return Tensor._from_vkbuf(vk.mul_add_relu(x_gpu._as_vkbuf(), y_gpu._as_vkbuf())).numpy()

    # Fused backend with preallocated output (no per-iter allocations).
    a_buf = x_gpu._as_vkbuf()
    b_buf = y_gpu._as_vkbuf()
    out_buf = vk.empty(a_buf.shape)

    def gpu_elemwise_fused_compute_only_no_alloc() -> None:
        vk.mul_add_relu_out(a_buf, b_buf, out_buf)

    cpu_times = _bench(cpu_elemwise, warmup, iters)
    gpu_rw_times = _bench(gpu_elemwise_with_readback, warmup, iters)
    gpu_co_times = _bench(gpu_elemwise_compute_only, warmup, iters)
    gpu_fused_rw_times = _bench(gpu_elemwise_fused_with_readback, warmup, iters)
    gpu_fused_co_times = _bench(gpu_elemwise_fused_compute_only, warmup, iters)
    gpu_fused_no_alloc_times = _bench(gpu_elemwise_fused_compute_only_no_alloc, warmup, iters)

    # One-time download timing (after compute-only benchmark)
    t0 = time.perf_counter_ns()
    _ = (z_hold.numpy() if z_hold is not None else None)
    t1 = time.perf_counter_ns()
    download_ms = (t1 - t0) / 1e6

    cpu_s = _stats_ms(cpu_times)
    gpu_rw_s = _stats_ms(gpu_rw_times)
    gpu_co_s = _stats_ms(gpu_co_times)
    gpu_fused_rw_s = _stats_ms(gpu_fused_rw_times)
    gpu_fused_co_s = _stats_ms(gpu_fused_co_times)
    gpu_fused_no_alloc_s = _stats_ms(gpu_fused_no_alloc_times)
    speedup_rw = cpu_s["mean_ms"] / gpu_rw_s["mean_ms"] if gpu_rw_s["mean_ms"] > 0 else float("inf")
    speedup_co = cpu_s["mean_ms"] / gpu_co_s["mean_ms"] if gpu_co_s["mean_ms"] > 0 else float("inf")
    speedup_fused_rw = (
        cpu_s["mean_ms"] / gpu_fused_rw_s["mean_ms"] if gpu_fused_rw_s["mean_ms"] > 0 else float("inf")
    )
    speedup_fused_co = (
        cpu_s["mean_ms"] / gpu_fused_co_s["mean_ms"] if gpu_fused_co_s["mean_ms"] > 0 else float("inf")
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
    print(_fmt_stats("GPU (Vulkan backend) fused compute-only", gpu_fused_co_s))
    print(f"speedup (mean, fused compute-only): {speedup_fused_co:.2f}x")
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

    def gpu_matmul_with_readback() -> np.ndarray:
        return (a_gpu @ b_gpu).numpy()

    c_hold: Tensor | None = None

    def gpu_matmul_compute_only() -> None:
        nonlocal c_hold
        c_hold = a_gpu @ b_gpu

    # Matmul with preallocated output (no per-iter allocations).
    a_buf = a_gpu._as_vkbuf()
    b_buf = b_gpu._as_vkbuf()
    c_buf = vk.empty((m, p))

    def gpu_matmul_compute_only_no_alloc() -> None:
        vk.matmul_out(a_buf, b_buf, c_buf)

    cpu_times = _bench(cpu_matmul, warmup, iters)
    gpu_rw_times = _bench(gpu_matmul_with_readback, warmup, iters)
    gpu_co_times = _bench(gpu_matmul_compute_only, warmup, iters)
    gpu_no_alloc_times = _bench(gpu_matmul_compute_only_no_alloc, warmup, iters)

    t0 = time.perf_counter_ns()
    _ = (c_hold.numpy() if c_hold is not None else None)
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
    # Simple elementwise ops on the "gpu" device (currently backed by NumPy).
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
    run_gpu_demo()
    run_benchmarks()
