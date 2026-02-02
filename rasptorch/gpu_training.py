from __future__ import annotations

import numpy as np

from . import vulkan_backend as vk
from .data import DataLoader, TensorDataset


class GpuMLP:
    def __init__(self, in_features: int, hidden: int, out_features: int, *, seed: int = 0) -> None:
        self.in_features = int(in_features)
        self.hidden = int(hidden)
        self.out_features = int(out_features)
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

        # Cache per-batch buffers to avoid per-iteration alloc/free churn.
        self._batch_cache: dict[int, dict[str, vk.VulkanBuffer]] = {}

        # Gradient buffers (shapes independent of batch).
        self._grad_w2 = vk.empty((self.hidden, self.out_features))
        self._grad_b2 = vk.empty((self.out_features,))
        self._grad_w1 = vk.empty((self.in_features, self.hidden))
        self._grad_b1 = vk.empty((self.hidden,))

    def _get_batch_buffers(self, batch: int) -> dict[str, vk.VulkanBuffer]:
        batch = int(batch)
        bufs = self._batch_cache.get(batch)
        if bufs is not None:
            return bufs

        bufs = {
            "x": vk.empty((batch, self.in_features)),
            "y_true": vk.empty((batch, self.out_features)),
            "z1": vk.empty((batch, self.hidden)),
            "z1b": vk.empty((batch, self.hidden)),
            "a1": vk.empty((batch, self.hidden)),
            "z2": vk.empty((batch, self.out_features)),
            "y_pred": vk.empty((batch, self.out_features)),
            "dY": vk.empty((batch, self.out_features)),
            "dA1": vk.empty((batch, self.hidden)),
            "dZ1": vk.empty((batch, self.hidden)),
        }
        self._batch_cache[batch] = bufs
        return bufs

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "w1": vk.to_cpu(self.w1),
            "b1": vk.to_cpu(self.b1),
            "w2": vk.to_cpu(self.w2),
            "b2": vk.to_cpu(self.b2),
        }

    def save(self, path: str) -> None:
        payload = {
            "arch": "GpuMLP",
            "in_features": self.in_features,
            "hidden": self.hidden,
            "out_features": self.out_features,
            "state_dict": self.state_dict(),
        }
        try:
            import torch  # type: ignore

            torch.save(payload, path)
        except ModuleNotFoundError:
            import pickle

            with open(path, "wb") as f:
                pickle.dump(payload, f)
            print("Warning: 'torch' not installed; wrote a pickle file (not torch.load compatible).")

    def close(self) -> None:
        # Free parameter buffers.
        vk.free(self.w1)
        vk.free(self.b1)
        vk.free(self.w2)
        vk.free(self.b2)

        # Free cached batch buffers.
        for bufs in self._batch_cache.values():
            for buf in bufs.values():
                vk.free(buf)
        self._batch_cache.clear()

        # Free reusable gradient buffers.
        vk.free(self._grad_w2)
        vk.free(self._grad_b2)
        vk.free(self._grad_w1)
        vk.free(self._grad_b1)

    def train_step(self, x_np: np.ndarray, y_np: np.ndarray, *, lr: float) -> float:
        x_np = np.asarray(x_np, dtype=np.float32)
        y_np = np.asarray(y_np, dtype=np.float32)
        bufs = self._get_batch_buffers(x_np.shape[0])

        # Upload batch into preallocated buffers.
        vk.write(bufs["x"], x_np)
        vk.write(bufs["y_true"], y_np)

        vk.begin_batch()
        # Forward
        vk.matmul_out(bufs["x"], self.w1, bufs["z1"])
        vk.add_rowvec_out(bufs["z1"], self.b1, bufs["z1b"])
        vk.relu_out(bufs["z1b"], bufs["a1"])
        vk.matmul_out(bufs["a1"], self.w2, bufs["z2"])
        vk.add_rowvec_out(bufs["z2"], self.b2, bufs["y_pred"])
        vk.end_batch()

        # Loss (CPU readback for logging only)
        pred_cpu = vk.to_cpu(bufs["y_pred"])
        loss = float(np.mean((pred_cpu - y_np) ** 2))

        vk.begin_batch()
        # Backward (explicit kernels)
        vk.mse_grad_out(bufs["y_pred"], bufs["y_true"], bufs["dY"])  # [batch, out]

        vk.matmul_at_b_out(bufs["a1"], bufs["dY"], self._grad_w2)  # [hidden, out]
        vk.reduce_sum_rows_out(bufs["dY"], self._grad_b2)  # [out]

        vk.matmul_a_bt_out(bufs["dY"], self.w2, bufs["dA1"])  # [batch, hidden]
        vk.relu_backward_out(bufs["dA1"], bufs["z1b"], bufs["dZ1"])  # [batch, hidden]

        vk.matmul_at_b_out(bufs["x"], bufs["dZ1"], self._grad_w1)  # [in, hidden]
        vk.reduce_sum_rows_out(bufs["dZ1"], self._grad_b1)  # [hidden]

        # SGD update (in-place)
        vk.sgd_update_inplace(self.w2, self._grad_w2, lr)
        vk.sgd_update_inplace(self.b2, self._grad_b2, lr)
        vk.sgd_update_inplace(self.w1, self._grad_w1, lr)
        vk.sgd_update_inplace(self.b1, self._grad_b1, lr)
        vk.end_batch()

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
    save_path: str | None = None,
) -> None:
    """Train a 2-layer MLP on GPU using explicit Vulkan backward kernels.

    Notes:
    - Forward/backward/update are done on GPU.
    - Loss is computed via CPU readback for logging.
    """

    # GPU mode must use Vulkan; fail fast with a clear error if unavailable.
    vk.init(strict=True)
    print("GPU training backend: Vulkan")

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

        if save_path:
            model.save(save_path)
            print(f"Saved model to: {save_path}")
    finally:
        model.close()
