from __future__ import annotations

import numpy as np

from . import vulkan_backend as vk
from .data import DataLoader, TensorDataset


class GpuMLP:
    def __init__(self, in_features: int, hidden: int, out_features: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)

        # Store weights in matmul-friendly layout:
        # W1: [in, hidden], W2: [hidden, out]
        w1_scale = np.sqrt(2.0 / max(1, in_features))
        w2_scale = np.sqrt(2.0 / max(1, hidden))

        w1 = (rng.standard_normal((in_features, hidden)).astype(np.float32) * w1_scale)
        b1 = np.zeros((hidden,), dtype=np.float32)
        w2 = (rng.standard_normal((hidden, out_features)).astype(np.float32) * w2_scale)
        b2 = np.zeros((out_features,), dtype=np.float32)

        self.w1 = vk.to_gpu(w1)
        self.b1 = vk.to_gpu(b1)
        self.w2 = vk.to_gpu(w2)
        self.b2 = vk.to_gpu(b2)

    def close(self) -> None:
        # Free parameter buffers.
        vk.free(self.w1)
        vk.free(self.b1)
        vk.free(self.w2)
        vk.free(self.b2)

    def train_step(self, x_np: np.ndarray, y_np: np.ndarray, *, lr: float) -> float:
        # Upload batch.
        x = vk.to_gpu(np.asarray(x_np, dtype=np.float32))
        y_true = vk.to_gpu(np.asarray(y_np, dtype=np.float32))

        # Forward
        z1 = vk.matmul(x, self.w1)
        z1b = vk.add_rowvec(z1, self.b1)
        a1 = vk.relu(z1b)
        z2 = vk.matmul(a1, self.w2)
        y_pred = vk.add_rowvec(z2, self.b2)

        # Loss (CPU readback for logging only)
        pred_cpu = vk.to_cpu(y_pred)
        loss = float(np.mean((pred_cpu - y_np) ** 2))

        # Backward (explicit kernels)
        dY = vk.mse_grad(y_pred, y_true)  # [batch, out]

        grad_w2 = vk.matmul_at_b(a1, dY)  # [hidden, out]
        grad_b2 = vk.reduce_sum_rows(dY)  # [out]

        dA1 = vk.matmul_a_bt(dY, self.w2)  # [batch, hidden]
        dZ1 = vk.relu_backward(dA1, z1b)  # [batch, hidden]

        grad_w1 = vk.matmul_at_b(x, dZ1)  # [in, hidden]
        grad_b1 = vk.reduce_sum_rows(dZ1)  # [hidden]

        # SGD update (in-place)
        vk.sgd_update_inplace(self.w2, grad_w2, lr)
        vk.sgd_update_inplace(self.b2, grad_b2, lr)
        vk.sgd_update_inplace(self.w1, grad_w1, lr)
        vk.sgd_update_inplace(self.b1, grad_b1, lr)

        # Free intermediates
        for buf in [
            x,
            y_true,
            z1,
            z1b,
            a1,
            z2,
            y_pred,
            dY,
            grad_w2,
            grad_b2,
            dA1,
            dZ1,
            grad_w1,
            grad_b1,
        ]:
            vk.free(buf)

        return loss


def train_mlp_regression_gpu(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.1,
    hidden: int = 16,
    seed: int = 0,
    log_every: int = 10,
) -> None:
    """Train a 2-layer MLP on GPU using explicit Vulkan backward kernels.

    Notes:
    - Forward/backward/update are done on GPU.
    - Loss is computed via CPU readback for logging.
    """

    vk.init()

    probe = vk.to_gpu(np.zeros((1,), dtype=np.float32))
    using_vulkan = probe.host is None and probe.buffer != 0
    vk.free(probe)

    print(f"GPU training backend: {'Vulkan' if using_vulkan else 'NumPy fallback'}")

    dataset = TensorDataset(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GpuMLP(in_features=x.shape[1], hidden=hidden, out_features=y.shape[1], seed=seed)
    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for xb_np, yb_np in loader:
                loss = model.train_step(xb_np, yb_np, lr=lr)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            if epoch % log_every == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.6f}")
    finally:
        model.close()
