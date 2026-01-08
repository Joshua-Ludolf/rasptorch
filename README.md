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

- Run the CPU training demo: `uv run main.py`
- Run GPU demos + benchmarks: `uv run gpu_demo.py`

## Benchmarks

`gpu_demo.py` prints timing stats (min/p50/p95/mean/std) for:

- CPU (NumPy)
- GPU compute+readback (includes `.numpy()` every iteration)
- GPU compute-only (no per-iteration readback)
- GPU fused compute-only and **no-alloc** variants (preallocated output buffers)

If you want the GPU to win, focus on the compute-only + fused/no-alloc numbers.

## Current Limitations

- Autograd is CPU-only. Calling `backward()` on a GPU tensor is not implemented.
- The GPU path currently targets a small set of ops; more kernels and/or additional
  fusions are needed for end-to-end training on GPU.
