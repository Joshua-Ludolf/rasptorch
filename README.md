# 🍓🎇 rasptorch

rasptorch is an experimental deep learning library inspired by PyTorch, built with a singular focus: **making complex neural networks practical and efficient to run on resource-constrained hardware like the Raspberry Pi 5, by leveraging its GPU capabilities via Vulkan.**

---

### ✨ Core Concepts

The library operates on a multi-layered architecture to maximize hardware utilization:

1.  **CPU Backend (Software):** Uses a pure NumPy-backed autograd engine and `nn` module for reliable computation when GPU acceleration is unavailable.
2.  **GPU Backend (Hardware):** Features an experimental **Vulkan backend** for high-speed tensor operations (elementwise math, matmul, reductions) directly on the Pi 5's GPU.
3.  **Interface:** Provides a streamlined CLI/Streamlit UI for interactive model building, training, persistence, and inspection.

The Vulkan path relies on real compute shaders compiled to SPIR-V, giving deep control over the underlying hardware.

### 🔌 Backend Abstraction (Connectable Backends)

rasptorch now exposes a backend abstraction API so compute backends can be registered and connected at runtime:

```python
import rasptorch

# Inspect availability
print(rasptorch.available_backends())  # {'cpu': True, 'vulkan': ..., 'opencl': ..., 'cuda': ...}

# Try to connect a backend (falls back to CPU in non-strict mode)
active = rasptorch.connect_backend("vulkan", strict=False)
print(active.name)
```

Built-in backend adapters:
- `numpy` (NumPy adapter; internal key: `cpu`) - Pure NumPy autograd
- `vulkan` (rasptorch Vulkan kernels, with optional CPU fallback) - **Optimized for Raspberry Pi 4/5** ⚡
- `opencl` (pyopencl when available, optional CPU fallback)
- `cuda` (CuPy when available, with PyTorch CUDA fallback, optional CPU fallback)

CLI helpers:
```bash
rasptorch backend list
rasptorch backend connect numpy
rasptorch backend connect vulkan --strict
# Benchmark with auto-tuned Vulkan kernel and submission batching
rasptorch --json backend benchmark --backends numpy,vulkan,cuda --size 2048 --iterations 100 --warmup 20 --vulkan-kernel auto --vulkan-autotune-submit --seed 42
```

> **Note:** User-facing CLI/UI labels the CPU backend as **`numpy`**.
> Vulkan benchmark mode uses resident buffers (upload once, repeated on-device matmul, download once).
> **Performance (Optimized):** Vulkan achieves ~564 GFLOPS (78% of NumPy on matmul_vec4 with auto-tuning).
> For very large square workloads (for example, 4096x4096), the benchmark now prefers the stable `matmul_vec4_wide_tiled` path by default.
> If you want to pin that path manually, use `--vulkan-kernel matmul_vec4_wide_tiled --vulkan-submit-every 8`.
> `--vulkan-kernel auto` probes `matmul_vec4_wide_tiled`, `matmul`, `matmul_vec4`, `matmul_vec4_tiled`, `matmul_a_bt`, and `matmul_a_bt_tiled` (when available) and keeps the faster path.
> If Vulkan hits `VkErrorDeviceLost`, lower `--vulkan-submit-every` (for example, `4` or `1`) or use auto-tuning.
> **Recommended:** Use `--vulkan-autotune-submit` to jointly probe kernel + submit chunk and pick the fastest stable combo.
> Optimizations: Command buffer batching, memory-mapped buffers, auto kernel selection.

### 📚 What's Included (Core Features)

*   **Tensor Operations:** Support for elementwise math, matrix multiplication (`matmul`), reductions, indexing, reshaping, stacking, and broadcasting.
*   **Layers:** Includes standard neural network blocks: `Linear`, `MLP`, `CNN`, `GRU`, `Transformer`, normalization layers, activations, pooling, embeddings, and attention.
*   **Training Tools:** Full suite of tools including optimizers (`SGD`), learning-rate schedulers, gradient clipping, and regularization helpers.
*   **Persistence:** Ability to save and load checkpoint weights without needing the full `torch` dependency.
*   **Interfaces:** CLI (`rasptorch chat`) and Streamlit UI (`rasptorch ui`).

### 🚀 Getting Started

#### 1. Installation

**A. Basic Install (CPU Only):**
To get the core library components running on the CPU:
```bash
pip install rasptorch
```

**B. Development Install (Full Capability):**
For local development and access to all potential backends:
```bash
pip install -e ".[dev]"
```

**C. GPU Mode Prerequisites:**
To utilize the GPU backend, you must meet these prerequisites:
**D. GPU Backend Dependencies:**
For systems with a proper GPU setup (e.g., Raspberry Pi 5), install the GPU backend dependencies using:
```bash

# Prerequisites for Vulkan backend (Raspberry Pi 4/5 & Linux)
sudo apt update
sudo apt install -y glslc (for raspberry pi 4/5 & linux operating systems)

# For Windows, install the Vulkan SDK from LunarG:
./VulkanSDK-Installer.exe --accept-licenses --default-answer --confirm-command install

# Then install rasptorch with GPU support after installing prerequisites (for either platform):
uv pip install rasptorch[gpu]
```

#### 2. Quick Run Examples

**Start the Interactive Shell:**
```bash
uv run rasptorch chat
```

**Launch the Web UI:**
```bash
uv run rasptorch ui
```
*(This will usually open at `http://localhost:8501`)*

**Viewing Help:**
To see all available CLI subcommands:
```bash
uv run rasptorch --help
```

---

### ⚙️ Execution Modes & Workflows

The `main.py` script controls the operational mode:

*   **`cpu`**: Pure NumPy autograd execution on the CPU.
*   **`gpu`**: Executes the training loop explicitly using the Vulkan backend kernels.
*   **`gpu-autograd`**: An experimental mode for tracing gradients across the GPU pipeline.

**Example Training Command:**
```bash
uv run main.py --device gpu --epochs 50 --batch-size 32 --lr 0.01
```

---

### 📊 Benchmarks

rasptorch provides a built-in benchmark tool for comparing backend performance on matrix multiplication:

**Quick Benchmark (Single Size):**
```bash
# Benchmark with default settings (2048x2048 matmul, 100 iterations)
uv run rasptorch backend benchmark

# Benchmark with custom size and multiple backends
uv run rasptorch --json backend benchmark --backends numpy,vulkan,cuda --size 2048 --iterations 100 --warmup 20 --seed 42
```

**Performance Results (Windows 11, NVIDIA GeForce RTX 3070 Ti):**

Measured backend scaling with:
```bash
uv run rasptorch --json backend benchmark --backends numpy,opencl,vulkan,cuda --vulkan-kernel matmul_vec4_wide_tiled --vulkan-submit-every 8 --iterations 5 --warmup 1 --seed 42
```

#### 256×256 → 4096×4096 Scaling
| Backend  | 256×256 (GFLOPS) | 512×512 (GFLOPS) | 1024×1024 (GFLOPS) | 2048×2048 (GFLOPS) | 4096×4096 (GFLOPS) |
|----------|------------------|------------------|-------------------|-------------------|-------------------|
| NumPy | 101.01 | 368.14 | 720.12 | 816.57 | 883.75 |
| OpenCL | 44.24 | 68.20 | 107.01 | 97.75 | 46.87 |
| Vulkan (`matmul_vec4_wide_tiled`) | 259.43 | 654.88 | 1043.37 | 809.62 | 534.92 |
| CUDA | 116.48 | 380.50 | 1063.82 | 2136.75 | 3990.81 |

#### Vulkan Kernel Comparison (2048×2048)
Measured with:
```bash
uv run rasptorch --json backend benchmark --backends vulkan --vulkan-submit-every 8 --size 2048 --iterations 5 --warmup 1 --seed 42 --vulkan-kernel <kernel>
```

| Vulkan Kernel | GFLOPS | Status |
|---------------|--------|--------|
| `matmul` | 492.73 | ok |
| `matmul_tiled` | 497.81 | ok |
| `matmul_vec4` | 493.15 | ok |
| `matmul_vec4_tiled` | 484.03 | ok |
| `matmul_vec4_wide_tiled` | 778.27 | ok |
| `matmul_a_bt` | 150.90 | ok |
| `matmul_a_bt_tiled` | 173.23 | ok |


**Vulkan Kernel Selection:**
The `--vulkan-kernel auto` flag intelligently probes available kernels:
- `matmul` - Basic single-threaded implementation
- `matmul_vec4` - SIMD-style vec4 operations
- `matmul_vec4_tiled` - vec4 plus shared-memory tiling
- `matmul_vec4_wide_tiled` - vec4 plus wider output tiling for better reuse
- `matmul_a_bt` - Matrix transpose optimization (for A @ B.T)
- `matmul_a_bt_tiled` - Tiled transpose optimization (fastest when applicable)

**Advanced Tuning:**
```bash
# Auto-tune both kernel AND submission batching strategy
uv run rasptorch --json backend benchmark --backends vulkan --size 2048 \
  --iterations 100 --warmup 20 \
  --vulkan-kernel auto \
  --vulkan-autotune-submit \
  --seed 42

# Manual kernel selection with custom batch submission
uv run rasptorch --json backend benchmark --backends vulkan \
  --vulkan-kernel matmul_a_bt_tiled \
  --vulkan-submit-every 4 \
  --size 2048 --iterations 100
```

**Large Workload Guidance:**
- For 4096x4096 matmul, use the default auto mode or pin `matmul_vec4_wide_tiled` directly.
- If a custom Vulkan kernel triggers `VkErrorDeviceLost`, reduce `--vulkan-submit-every` first.
- The benchmark path uses resident buffers, so the large-size slowdown is kernel quality rather than repeated host-device transfers.

**Output Format:**
Results are provided in JSON format (with `--json` flag) including:
- `status`: "ok" or "unavailable"
- `elapsed_seconds`: Total benchmark time
- `iterations_per_second`: Throughput metric
- `estimated_gflops`: Floating-point performance
- `checksum`: Verification result
- `kernel`: Selected kernel name (for auto mode)
- `submit_every`: Submission batch size (for Vulkan)

**Optimization Tips:**
- Use `--vulkan-autotune-submit` for best results (probes kernel + batch combinations)
- If you see `VkErrorDeviceLost`, reduce `--vulkan-submit-every` (try `4` or `1`)
- Larger problem sizes better amortize GPU setup overhead
- Command buffer batching (`--vulkan-submit-every`) balances latency and throughput

For detailed optimization guide, see [VULKAN_OPTIMIZATION.md](VULKAN_OPTIMIZATION.md).

---

### 🧠 Advanced Topics

#### 1. Tensor Operations
Basic tensor math is performed via:
```bash
# Create tensors
uv run rasptorch tensor random --shape 2,3,4
uv run rasptorch tensor ones --shape 5,10
```
The results show the low-level tensor capabilities.

#### 2. Model Definition
Models are defined using structured commands:
```bash
# Simple MLP
uv run rasptorch model mlp --layers "64,32,16,2"
# Complex CNN
uv run rasptorch model cnn --in-channels 3 --out-channels "32,64,128"
```
Managing the lifecycle:
```bash
uv run rasptorch model list
uv run rasptorch model save --model-id <id> --path model.pth
```

---

### 🩹 Troubleshooting & Best Practices

1.  **Performance:** The fastest paths are those that keep the computation entirely on the GPU and minimize data transfer across the PCIe bus.
2.  **Fallback:** If GPU operations fail due to driver issues, the system gracefully falls back to the CPU NumPy path, but performance will suffer.
3.  **Advanced Use:** For understanding the deep dive into custom kernel optimization, please refer to the source code in the `rasptorch/gpu_demo.py` and `rasptorch/main.py` scripts.
