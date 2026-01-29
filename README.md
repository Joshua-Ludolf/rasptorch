## rasptorch

rasptorch is an experimental deep learning library inspired by PyTorch,
with a specific goal: **make training and running neural networks
practical on the Raspberry Pi 5 while taking advantage of its GPU**.

The project has two main parts:

- A small, NumPy-backed autograd engine and `nn` module that runs on the
	Raspberry Pi CPU.
- An experimental Vulkan-based backend, wired through a `device` API
	(`Tensor(..., device="cpu"|"gpu").to("gpu")`), meant to offload core
	tensor operations (elementwise ops, matmul, activations) to the Pi 5's
	GPU via Vulkan compute.

The Vulkan backend is implemented with real Vulkan compute shaders (GLSL
compiled to SPIR-V). It supports a small but useful set of kernels:

- Elementwise: `add`, `mul`, `relu`
- Matmul: `matmul` (tiled/shared-memory shader)
- Fused kernel: `mul_add_relu` for `(x * y + x).relu()`

Performance notes:

- The fastest path is **compute-only** (keep tensors on GPU, avoid per-iteration
  `.numpy()`/readbacks).
- Fusing ops and reusing output buffers can make the Vulkan path **faster than
  NumPy** for certain workloads on Raspberry Pi 5.

See `main.py` for a simple training example and `gpu_demo.py` for a
focused test + benchmark suite for the Vulkan backend.

## Quickstart

- Ensure you have `uv` installed in your python enviroment. See https://docs.astral.sh/uv/#installation to get started with it.

## Installation

From PyPI:

- `pip install rasptorch`

Notes for GPU mode:

- Requires working Vulkan drivers on your system.
- Requires `glslc` (shader compiler) available on PATH.

For local development from this repo:

- `pip install -e .`

## GPU Training (Vulkan)

There are currently two “modes” of training in this repo:

- **CPU autograd training** (PyTorch-like): uses the NumPy-backed autograd engine.
- **Vulkan GPU training** (explicit kernels): runs forward + backward + SGD updates on GPU
	using purpose-built compute shaders.

The Vulkan training path lives in `rasptorch/gpu_training.py` and currently supports a
2-layer MLP:

`Linear -> ReLU -> Linear` with MSE loss and SGD.

Run it via:

- `uv run main.py --device gpu --epochs 50 --batch-size 32 --lr 0.1`

## GPU Autograd (WIP)

There is now an experimental **gpu-autograd** mode that enables `loss.backward()` even when
the model and activations live on GPU, for a limited set of ops.

Run it via:

- `uv run main.py --device gpu-autograd --epochs 50 --batch-size 32 --lr 0.1`

Currently supported (GPU) in autograd:

- `+`, `*`, `-` (scalar and tensor forms), `@` (matmul)
- scalar ops: `tensor + s`, `tensor * s`, `tensor / s`, plus `s + tensor`, `s * tensor`, `s - tensor`
- `neg`, `relu`, `sum`, `mean`, `T` (2D transpose)
- `Linear` backward (GPU grads for `weight`/`bias`)
- `SGD.step()` updates GPU parameters in-place (SGD + optional momentum/weight decay)
- `functional.cross_entropy(logits, target_onehot)` (softmax cross-entropy, mean reduction)

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
- There is not yet a `no_grad()` context; evaluation still builds graphs.

- `mse_loss` is now implemented purely via tensor ops (`(pred-target)^2` + `mean()`), so the
loss tensor itself is on GPU in `gpu-autograd` mode; training code typically reads it back
via `.numpy()` for logging.

- Parameters and gradients stay on GPU; **loss is read back to CPU for logging**.
- `uv run main.py --device gpu` now **requires Vulkan**. If Vulkan init or shader compilation fails, it raises a clear error instead of silently falling back.
- Broadcasting is still limited; common 2D + 1D row-vector forms like `(N,M) + (M,)` and `(N,M) * (M,)` are supported.

## Benchmarks

`gpu_demo.py` prints timing stats (min/p50/p95/mean/std) for:

- CPU (NumPy)
- GPU compute+readback (includes `.numpy()` every iteration)
- GPU compute-only (no per-iteration readback)
- GPU fused compute-only and **no-alloc** variants (preallocated output buffers)

If you want the GPU to win, focus on the compute-only + fused/no-alloc numbers.

## Current Limitations

- GPU autograd is still **incomplete** (only a subset of ops are supported).
- GPU reductions now support `sum()` and `mean()`, but other reductions/broadcast patterns
	are still limited.
- The Vulkan backend only implements a small set of ops/kernels; expanding model coverage will
	require more kernels (and ideally more fusion).
- PyTorch integration is experimental: `rasptorch.torch_bridge` currently supports a small inference subset
  (Conv2d/Linear/ReLU) and may copy tensors CPU<->GPU.

## Publishing (maintainers)

Build:

- `python -m pip install -U build twine`
- `python -m build`

Upload PyPI:

- `python -m twine upload dist/*`
