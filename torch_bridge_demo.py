"""Demo: wrap a small torch CNN and run conv/linear/relu via rasptorch Vulkan.

This is inference-focused and allows CPU fallback for unsupported ops.

Usage:
  python torch_bridge_demo.py

Requires:
  pip install torch
  Vulkan drivers + glslc (for GPU mode)
"""

from __future__ import annotations

import numpy as np

import rasptorch.vulkan_backend as vk
from rasptorch.torch_bridge import convert_torch_model


def main() -> None:
    import torch

    torch.manual_seed(0)

    # Small "torchvision-ish" CNN subset (no pooling yet).
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 16 * 16, 10),
    ).eval()

    x = torch.from_numpy(np.random.randn(1, 3, 32, 32).astype(np.float32))

    # Baseline CPU
    with torch.no_grad():
        y_cpu = model(x)

    # Convert supported pieces
    model2 = convert_torch_model(model, device="gpu").eval()

    # Ensure Vulkan present for demo (fail-fast)
    vk.init(strict=True)

    with torch.no_grad():
        y_vk = model2(x)

    diff = (y_vk - y_cpu).abs().max().item()
    print("max|diff|:", diff)


if __name__ == "__main__":
    main()
