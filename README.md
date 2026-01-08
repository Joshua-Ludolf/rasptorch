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

Right now, the Vulkan backend is a stub that still uses NumPy under the
hood, but the interfaces are in place so you can progressively replace
them with real Vulkan code and benchmark GPU vs CPU performance on the
Pi 5.

See `main.py` for a simple training example and `gpu_demo.py` for a
focused test of the experimental GPU code path.
