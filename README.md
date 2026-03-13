# rasptorch

rasptorch is an experimental deep learning library inspired by PyTorch,
with a specific goal: **make training and running neural networks practical on the
Raspberry Pi 5 while taking advantage of its GPU**.

The project has two main parts:

- A small, NumPy-backed autograd engine and `nn` module that runs on the Raspberry Pi CPU.
- An experimental Vulkan-based backend, wired through a `device` API
  (`Tensor(..., device="cpu"|"gpu").to("gpu")`), meant to offload core tensor operations
  (elementwise ops, matmul, activations, reductions, etc.) to the Pi 5's GPU via Vulkan compute.

The Vulkan backend is implemented with real Vulkan compute shaders (GLSL
compiled to SPIR-V). It supports a small but useful set of kernels (see
`rasptorch/shaders/` for the authoritative list).

High-level highlights:

- Elementwise ops: `+`, `*`, `-`, `neg`, `relu`, `gelu`, `silu`, `leaky_relu`, `elu` (plus scalar variants)
- Matmul: `@` (tiled/shared-memory shader)
- Tensor helpers: indexing/slicing via `tensor[...]`, `unsqueeze`, `squeeze`, `permute`, `transpose`, `flatten`, `max`, `min`, `argmax`, `argmin`, `cat`, `stack`, `split`, `chunk`
- Reductions: `sum`, `mean` (global and axis-based on CPU; global on GPU)
- Common broadcast forms: `(N,M) + (M,)` and `(N,M) * (M,)`
- Losses: `cross_entropy`, `binary_cross_entropy`, `binary_cross_entropy_with_logits`, `nll_loss`, `smooth_l1_loss`, `label_smoothing_cross_entropy`
- Optimizers and training utilities: `SGD`, `Adam`, `AdamW`, `RMSProp`, LR schedulers, gradient clipping, regularization helpers
- NN essentials: GPU row-wise `softmax` / `log_softmax`, 2D `LayerNorm`, `BatchNorm1d`, `BatchNorm2d`, `Embedding`, `MultiheadAttention`, `MaxPool2d`, `AvgPool2d`, and `GRU`

Performance notes:

- The fastest path is **compute-only**: keep tensors on GPU and avoid per-iteration
  `.numpy()` / readbacks.
- Fusing ops and reusing output buffers can make the Vulkan path **faster than NumPy**
  for certain workloads on Raspberry Pi 5.

See `main.py` for a simple training example and `gpu_demo.py` for a
focused correctness + benchmark suite for the Vulkan backend.

## Demos

- Essentials demo (softmax/log_softmax, LayerNorm, Dropout, `no_grad`, `detach`):
  - CPU: `uv run essentials_demo.py --device cpu`
  - GPU-autograd (Vulkan): `uv run essentials_demo.py --device gpu`

Note: the Vulkan-backed GPU path requires working Vulkan drivers and `glslc` (shader compiler)
on your `PATH`. Low-level backend helpers can still fall back to NumPy in some environments,
but `main.py --device gpu` is expected to run with Vulkan (glslc available) and fails clearly if it is not.

## Modes

There are three execution modes exposed via `main.py --device ...`:

- `cpu`: NumPy autograd engine (PyTorch-like, runs on CPU)
- `gpu`: explicit Vulkan training path (forward + backward + SGD via purpose-built kernels)
- `gpu-autograd`: experimental GPU autograd (builds a graph on GPU for a growing but still incomplete set of ops)

## Quickstart

- Use a virtual environment for best results (e.g. `.venv`, `.venv + uv`, `.venv + poetry`).

## Installation

From PyPI (CPU-only):

- `pip install rasptorch`

GPU (Pi 5 Vulkan):

- `pip install "rasptorch[gpu]"`

Optional (for saving/loading `.pth` via real `torch.save`/`torch.load`):

- `pip install "rasptorch[torch]"`

Dev/test:

- `pip install -e ".[dev]"`

Notes for GPU mode:

- Requires working Vulkan drivers on your system.
- Requires `glslc` (shader compiler) available on PATH.

Quick GPU validation:

- `uv run gpu_demo.py --smoke-only`
  - Initializes Vulkan strictly and runs fast correctness checks for core kernels.
  - If this fails, `uv run main.py --device gpu` will also fail.

Quick model saving check:

- `uv run main.py --device cpu --epochs 1 --save model.pth`
  - If `torch` is installed: `python -c "import torch; print(torch.load('model.pth').keys())"`
  - If not: `python -c "import pickle; print(pickle.load(open('model.pth','rb')).keys())"`

For local development from this repo:

- `pip install -e .`


### CLI Quickstart

rasptorch includes an **agent-native CLI** for tensor operations, model management, and training:

```bash
# Show available commands
python rasptorch --help

# chat
python rasptorch chat (you can type help once the cli is up and running in chat mode)

# Create tensors
python rasptorch tensor random --shape 2,3,4
python rasptorch tensor zeros --shape 3,4

# Manage models
python rasptorch model list
python rasptorch model create-linear --input-size 10 --hidden-sizes "32,16" --output-size 2

# JSON output for scripting/agents
python rasptorch --json tensor random --shape 2,3,4
```

See [rasptorch CLI.md](rasptorch%20CLI.md) for **complete CLI documentation**, including training, saving/loading models, and agent integration.


## GPU Training (Vulkan)

There are currently two “modes” of training in this repo:

- **CPU autograd training** (PyTorch-like): uses the NumPy-backed autograd engine.
- **Vulkan GPU training** (explicit kernels): runs forward + backward + SGD updates on GPU
  using purpose-built compute shaders.

The Vulkan training path lives in `rasptorch/gpu_training.py` and currently supports a
2-layer MLP:

`Linear -> ReLU -> Linear` with MSE loss and SGD.

The general `gpu-autograd` path supports substantially more model-building pieces than the
explicit `gpu` trainer, including adaptive optimizers, additional activations, LayerNorm,
BatchNorm, Embedding, and MultiheadAttention.

Run it via:

- `uv run main.py --device gpu --epochs 50 --batch-size 32 --lr 0.1`

Saving weights (PyTorch-style `.pth`):

- `uv run main.py --device gpu --epochs 50 --save model.pth`
- If `torch` is installed, this is a real `torch.save(...)` file loadable via `torch.load("model.pth")`.
- If `torch` is not installed, rasptorch falls back to writing a pickle payload (same keys, **not** `torch.load` compatible).

## GPU Autograd (WIP)

There is now an experimental **gpu-autograd** mode that enables `loss.backward()` even when
the model and activations live on GPU, for a limited set of ops.

Run it via:

- `uv run main.py --device gpu-autograd --epochs 50 --batch-size 32 --lr 0.1`

Currently supported (GPU) in autograd:

- `+`, `*`, `-` (scalar and tensor forms), `@` (matmul)
- scalar ops: `tensor + s`, `tensor * s`, `tensor / s`, plus `s + tensor`, `s * tensor`, `s - tensor`
- `neg`, `relu`, `gelu`, `silu`, `leaky_relu`, `elu`, `sum`, `mean`, `T` (2D transpose)
- tensor shape/join helpers: `unsqueeze`, `squeeze`, `flatten`, `permute` (common tensors up to 4D), `transpose(dim0, dim1)`, `cat`, `stack`, `split`, `chunk`
- `functional.softmax` / `functional.log_softmax` (2D row-wise, `dim=-1/1`)
- `nn.LayerNorm` (2D inputs, 1D `normalized_shape`; `eps=1e-5` stays on GPU, other `eps` values fall back to CPU)
- `nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.Embedding`, `nn.MultiheadAttention`
- `Linear` backward (GPU grads for `weight`/`bias`)
- `SGD.step()` updates GPU parameters in-place (SGD + optional momentum/weight decay)
- `Adam.step()`, `AdamW.step()`, `RMSProp.step()` update GPU parameters in-place
- `functional.cross_entropy(logits, target_onehot)` (softmax cross-entropy, mean reduction)

Also available on the CPU autograd path:

- `GRU` backward
- `Tensor.__getitem__` / slicing
- axis-based `sum(axis=...)` / `mean(axis=...)`
- `max(axis=...)` / `min(axis=...)` with autograd
- `argmax(axis=...)` / `argmin(axis=...)`
- `nn.MaxPool2d` / `nn.AvgPool2d`

Also available across the library:

- LR schedulers: `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, `WarmupScheduler`
- Initialization helpers: `kaiming_*`, `xavier_*`, `orthogonal_`, `uniform_`, `normal_`, `zeros_`, `ones_`, `constant_`
- Gradient utilities: `clip_grad_norm_`, `clip_grad_value_`, `l1_regularization`, `l2_regularization`, `total_variation_loss`
- AMP surface: `rasptorch.amp.autocast()` and `rasptorch.amp.GradScaler`

Tip: `rasptorch.no_grad()` exists (like PyTorch) to disable graph building during evaluation.

## Training Loop Utilities

There is now a small, reusable training loop helper in `rasptorch.train` that provides
PyTorch-like epoch logs (loss, accuracy/metrics, throughput) for **any** model.

Key pieces:

- `rasptorch.train.fit(...)`: train loop with optional validation
- `rasptorch.train.Accuracy()`: top-1 classification accuracy
- `rasptorch.train.classification_target_one_hot(C, device=...)`: converts integer labels -> one-hot

Example (classifier):

```python
from rasptorch import functional as F
from rasptorch.train import fit, Accuracy, classification_target_one_hot
from rasptorch.optim import SGD

model = ...
opt = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

fit(
    model,
    opt,
    train_loader,
    loss_fn=F.cross_entropy,
    device="gpu",
    epochs=10,
    val_loader=val_loader,
    target_transform=classification_target_one_hot(num_classes=10, device="gpu"),
    metrics=[Accuracy()],
)
```

Notes:

- Metrics like accuracy call `.numpy()` on logits, which triggers a GPU readback.
- `rasptorch.no_grad()` exists; evaluation can avoid building graphs.
- `mse_loss` is now implemented purely via tensor ops (`(pred-target)^2` + `mean()`), so the
  loss tensor itself is on GPU in `gpu-autograd` mode; training code typically reads it back
  via `.numpy()` for logging.
- Parameters and gradients stay on GPU; **loss is read back to CPU for logging**.
- `uv run main.py --device gpu` now **requires Vulkan**. If Vulkan init or shader compilation fails, it raises a clear error instead of silently falling back.
- Broadcasting is still limited; common 2D + 1D row-vector forms like `(N,M) + (M,)` and `(N,M) * (M,)` are supported.

## Additional APIs

Optimization:

- `rasptorch.optim`: `SGD`, `Adam`, `AdamW`, `RMSProp`
- `rasptorch.optim_sched`: `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, `WarmupScheduler`

Initialization:

- `rasptorch.init`: `kaiming_uniform_`, `kaiming_normal_`, `xavier_uniform_`, `xavier_normal_`, `orthogonal_`, `constant_`, `zeros_`, `ones_`, `uniform_`, `normal_`

Regularization and gradient helpers:

- `rasptorch.utils`: `clip_grad_norm_`, `clip_grad_value_`, `l1_regularization`, `l2_regularization`, `total_variation_loss`

Tensor helpers:

- `Tensor.unsqueeze()`, `Tensor.squeeze()`, `Tensor.permute()`, `Tensor.transpose()`, `Tensor.flatten()`
- `Tensor.split()`, `Tensor.chunk()`
- `rasptorch.cat(...)`, `rasptorch.stack(...)`

GPU notes for tensor helpers:

- `unsqueeze`, `squeeze`, and `flatten` are view-based on GPU
- `cat`, `stack`, `split`, and `chunk` now use Vulkan device-to-device buffer copies
- `permute` / general `transpose(dim0, dim1)` are Vulkan-native for common tensors up to 4D

More modules:

- `rasptorch.nn`: `BatchNorm1d`, `BatchNorm2d`, `Embedding`, `MultiheadAttention`, `GRU`, `MaxPool2d`, `AvgPool2d`, `GELU`, `SiLU`, `LeakyReLU`, `ELU`

Mixed precision surface:

- `rasptorch.amp.autocast()`
- `rasptorch.amp.GradScaler`
- `Tensor.half()` / `Tensor.float()`

## Benchmarks

`gpu_demo.py` prints timing stats (min/p50/p95/mean/std) for:

- CPU (NumPy)
- GPU compute+readback (includes `.numpy()` every iteration)
- GPU compute-only (no per-iteration readback)
- GPU fused compute-only and **no-alloc** variants (preallocated output buffers)

If you want the GPU to win, focus on the compute-only + fused/no-alloc numbers.

## Current Limitations

- GPU autograd is still **incomplete**. Core MLP/classification paths are covered, but full PyTorch-like operator coverage is not there yet.
- Some newer APIs use CPU-backed math internally when no dedicated Vulkan/autograd kernel exists yet. The public API works, but not every path is fully GPU-native.
- The newer GPU-native tensor helper coverage is strongest for practical tensors up to 4D; generic higher-rank permutation is not on a dedicated Vulkan path yet.
- `GRU` autograd is currently CPU-backed; there is no dedicated Vulkan GRU autograd path yet.
- The mixed-precision API surface exists, but true fp16 Vulkan storage/compute kernels are not implemented yet. `autocast()` and `GradScaler` are currently preparatory/experimental.
- GPU reductions still focus on the common paths: global `sum()` / `mean()` are GPU-native, while axis-based reductions currently fall back to CPU.
- PyTorch integration is experimental: `rasptorch.torch_bridge` currently supports a small inference subset
  (`Conv2d`, `Linear`, `ReLU`, `BatchNorm2d`, `MaxPool2d`, `Sigmoid`, `Tanh`, `GELU`, `Dropout`) and may copy tensors CPU<->GPU.

## Development & Tests

- `pytest` runs CPU tests by default.
- The backend smoke test runs everywhere:
  - With Vulkan available, it exercises real GPU kernels.
  - Without Vulkan, it exercises the NumPy fallback path.
- For a strict Vulkan-only check, run `uv run gpu_demo.py --smoke-only`.

## Publishing (maintainers)

Build:

- `python -m pip install -U build twine`
- `python -m build`

Upload PyPI:

- `python -m twine upload dist/*`
