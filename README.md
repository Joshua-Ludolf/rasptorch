# rasptorch

rasptorch is an experimental deep learning library inspired by PyTorch, built around a simple goal: make neural networks practical on a Raspberry Pi 5 while taking advantage of its GPU.

It ships with three layers that work together:

1. A NumPy-backed autograd engine and `nn` module for CPU execution.
2. An experimental Vulkan backend for GPU tensor operations and GPU-focused training paths.
3. A chat-style CLI and Streamlit UI for interactive model building, training, persistence, and inspection.

## What’s Included

- Tensor ops: elementwise math, matmul, reductions, indexing, reshaping, stacking, and broadcasting.
- Neural network layers: `Linear`, `MLP`, `CNN`, `GRU`, `Transformer`, normalization layers, activations, pooling, embeddings, and attention.
- Training tools: optimizers, learning-rate schedulers, gradient clipping, regularization helpers, and reusable training loops.
- Persistence: save and load rasptorch checkpoints without requiring `torch`.
- Interfaces: `rasptorch chat` for REPL-style interaction and `rasptorch ui` for the Streamlit dashboard.

The Vulkan path uses real compute shaders compiled to SPIR-V. The supported kernels live in [`rasptorch/shaders/`](rasptorch/shaders/).

## Installation

### From PyPI

Base install:

```bash
pip install rasptorch
```

This installs the Python Vulkan bindings used by the GPU backend, but it does not by itself guarantee that the Pi 5 GPU path is available.

The optional `gpu` extra currently maps to the same Vulkan Python dependency.

Development install:

```bash
pip install -e ".[dev]"
```

### Requirements for GPU mode

- Raspberry Pi 5 with working Vulkan drivers.
- `glslc` on your `PATH`.
- Use the GPU-capable device path (`--device gpu` or `--device auto` when Vulkan is working).

If either of those is missing, GPU mode will fail clearly instead of silently pretending to work.

## Quick Start

Run the chat REPL:

```bash
uv run rasptorch chat
```

Launch the UI:

```bash
uv run rasptorch ui
```

Open the Streamlit app at `http://localhost:8501` unless you pass a different port.

Show the top-level CLI help:

```bash
uv run rasptorch --help
```

## Execution Modes

`main.py` exposes three training/runtime modes:

- `cpu`: NumPy autograd on the CPU.
- `gpu`: explicit Vulkan training with purpose-built kernels.
- `gpu-autograd`: experimental GPU autograd for a growing set of ops.

Examples:

```bash
uv run main.py --device cpu --epochs 10
uv run main.py --device gpu --epochs 50 --batch-size 32 --lr 0.1
uv run main.py --device gpu-autograd --epochs 50 --batch-size 32 --lr 0.1
```

## CLI Examples

Create tensors:

```bash
uv run rasptorch tensor random --shape 2,3,4
uv run rasptorch tensor zeros --shape 3,4
uv run rasptorch tensor ones --shape 5,10
```

Build models:

```bash
uv run rasptorch model linear --input-size 10 --hidden-sizes "32,16" --output-size 2
uv run rasptorch model mlp --layers "64,32,16,2"
uv run rasptorch model cnn --in-channels 3 --out-channels "32,64,128"
uv run rasptorch model transformer --vocab-size 1000 --d-model 128 --num-heads 4 --num-layers 2
```

Manage models:

```bash
uv run rasptorch model list
uv run rasptorch model remove --model-id <model-id>
uv run rasptorch model save --model-id <model-id> --path model.pth
uv run rasptorch model load --path model.pth
uv run rasptorch model combine <model-a> <model-b>
```

Train a model:

```bash
uv run rasptorch model train --model-id <model-id> --epochs 10 --lr 0.001 --batch-size 32
```

Use JSON output for scripting or agents:

```bash
uv run rasptorch --json tensor zeros --shape 3,4
```

See [`rasptorch/CLI/rasptorch CLI.md`](rasptorch/CLI/rasptorch%20CLI.md) for the full CLI reference.

## UI Overview

The Streamlit UI is organized around a few focused pages:

- Models: browse, inspect, load, save, delete, and visualize model structure.
- Build & Train: create models, combine models, run training, and launch hyperparameter search.
- Dashboard: compare training runs and loss curves.
- Chat/REPL: use the same chat-style commands as the terminal CLI.
- Logs: review session events and actions.

The UI also includes dataset validation and preprocessing controls, a persistent log viewer, and a lightweight explainability preview for uploaded images.

## GPU Notes

The fast path is compute-only: keep tensors on GPU and avoid unnecessary `.numpy()` readbacks.

Useful checks:

```bash
uv run gpu_demo.py --smoke-only
uv run main.py --device gpu --epochs 1 --save model.pth
```

The first command validates the Vulkan backend. The second produces a rasptorch checkpoint without needing `torch`.

## Training Utilities

`rasptorch.train` provides a small reusable training loop with optional validation and metrics.

Common pieces:

- `rasptorch.train.fit(...)`
- `rasptorch.train.Accuracy()`
- `rasptorch.train.classification_target_one_hot(...)`

Example:

```python
from rasptorch import functional as F
from rasptorch.optim import SGD
from rasptorch.train import Accuracy, classification_target_one_hot, fit

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

## PyTorch Bridge

`rasptorch.torch_bridge` can convert supported PyTorch inference models so compatible layers run on GPU and unsupported layers fall back to CPU.

```python
from rasptorch.torch_bridge import convert_torch_model
import torch

torch_model = torch.nn.Sequential(
    torch.nn.Linear(10, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10),
).eval()

rasp_model = convert_torch_model(torch_model, device="gpu")
```

## Project Layout

- [`rasptorch/`](rasptorch/) - core library modules.
- [`rasptorch/CLI/`](rasptorch/CLI/) - Click CLI and chat REPL.
- [`rasptorch/ui/app.py`](rasptorch/ui/app.py) - Streamlit UI.
- [`rasptorch/shaders/`](rasptorch/shaders/) - Vulkan compute shaders.
- [`tests/`](tests/) - CPU, CLI, UI, and backend tests.

## Development

Run the test suite:

```bash
pytest
```

Run the Vulkan smoke test:

```bash
uv run gpu_demo.py --smoke-only
```

Build packaging artifacts:

```bash
python -m pip install -U build twine
python -m build
```

## Limitations

- Vulkan support is experimental and focused on the Raspberry Pi 5.
- GPU autograd coverage is growing, but it is not complete.
- Some higher-level features still fall back to CPU paths depending on the operation and shape.

## License

MIT. See [`LICENSE`](LICENSE).
