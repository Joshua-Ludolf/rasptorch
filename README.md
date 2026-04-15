# 🦀 rasptorch 🦀

rasptorch is an experimental deep learning library inspired by PyTorch, built with a singular focus: **making complex neural networks practical and efficient to run on resource-constrained hardware like the Raspberry Pi 5, by leveraging its GPU capabilities via Vulkan.**

---

### ✨ Core Concepts

The library operates on a multi-layered architecture to maximize hardware utilization:

1.  **CPU Backend (Software):** Uses a pure NumPy-backed autograd engine and `nn` module for reliable computation when GPU acceleration is unavailable.
2.  **GPU Backend (Hardware):** Features an experimental **Vulkan backend** for high-speed tensor operations (elementwise math, matmul, reductions) directly on the Pi 5's GPU.
3.  **Interface:** Provides a streamlined CLI/Streamlit UI for interactive model building, training, persistence, and inspection.

The Vulkan path relies on real compute shaders compiled to SPIR-V, giving deep control over the underlying hardware.

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
*   Raspberry Pi 5 with working Vulkan drivers.
*   The `glslc` shader compiler must be available in your system `PATH`.
*   When running, you must specify the device: `--device gpu` or `--device auto`.

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
