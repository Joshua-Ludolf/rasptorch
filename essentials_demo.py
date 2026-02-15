from __future__ import annotations

import argparse

import numpy as np

import rasptorch
from rasptorch import Tensor
from rasptorch import functional as F
from rasptorch import vulkan_backend as vk
from rasptorch.nn import Dropout, LayerNorm, Linear, ReLU, Sequential
from rasptorch.optim import SGD


def _device_banner(device: str) -> str:
    if device != "gpu":
        return device
    vk.init(strict=False)
    reason = vk.disabled_reason()
    if reason:
        return f"gpu (NumPy fallback: {reason})"
    return "gpu (Vulkan)"


def _one_hot(labels: np.ndarray, num_classes: int, device: str) -> Tensor:
    oh = F.one_hot(labels.astype(np.int64), num_classes)
    return oh.to(device)


def demo_softmax_logsoftmax_backward(*, device: str) -> None:
    rng = np.random.default_rng(0)
    N, C = 8, 5

    logits = Tensor(rng.standard_normal((N, C), dtype=np.float32), requires_grad=True).to(device)
    labels = rng.integers(0, C, size=(N,), dtype=np.int64)
    target = _one_hot(labels, C, device)

    logp = F.log_softmax(logits, dim=1)
    loss = -(target * logp).sum() / float(N)

    loss.backward()

    loss_val = float(loss.numpy().reshape(-1)[0])
    grad_norm = 0.0
    if logits.grad is not None:
        grad_norm = float(np.linalg.norm(logits.grad))
    if logits.grad_vkbuf is not None:
        grad_norm = float(np.linalg.norm(vk.to_cpu(logits.grad_vkbuf)))

    print(f"[softmax/log_softmax] loss={loss_val:.6f} grad_norm={grad_norm:.6f}")


def demo_layernorm_dropout_mlp_training(*, device: str, steps: int = 5) -> None:
    rng = np.random.default_rng(1)
    N, Din, H, C = 32, 16, 32, 6

    x_np = rng.standard_normal((N, Din), dtype=np.float32)
    y = rng.integers(0, C, size=(N,), dtype=np.int64)

    x = Tensor(x_np, requires_grad=False).to(device)
    target = _one_hot(y, C, device)

    model = Sequential(
        Linear(Din, H),
        LayerNorm(H),
        ReLU(),
        Dropout(p=0.25),
        Linear(H, C),
    ).to(device)

    opt = SGD(model.parameters(), lr=0.3, momentum=0.0)

    model.train(True)
    losses: list[float] = []
    for _ in range(int(steps)):
        opt.zero_grad()

        logits = model(x)
        logp = F.log_softmax(logits, dim=1)
        loss = -(target * logp).sum() / float(N)

        loss.backward()
        opt.step()

        losses.append(float(loss.numpy().reshape(-1)[0]))

    print(f"[MLP+LayerNorm+Dropout] losses={', '.join(f'{v:.4f}' for v in losses)}")

    # Show dropout behavior: train() is stochastic, eval() is deterministic.
    model.train(True)
    a = model(x).numpy()
    b = model(x).numpy()
    train_diff = float(np.mean(np.abs(a - b)))

    model.eval()
    c = model(x).numpy()
    d = model(x).numpy()
    eval_diff = float(np.mean(np.abs(c - d)))

    print(f"[dropout] mean_abs_diff train={train_diff:.6f} eval={eval_diff:.6f}")


def demo_no_grad_and_detach(*, device: str) -> None:
    rng = np.random.default_rng(2)

    x = Tensor(rng.standard_normal((4, 4), dtype=np.float32), requires_grad=True).to(device)

    # detach(): should stop gradients.
    y = (x * 2.0).detach()
    z = (y * 3.0).sum()
    z.backward()

    has_grad = (x.grad is not None) or (x.grad_vkbuf is not None)
    print(f"[detach] x.grad is None? {not has_grad}")

    # no_grad(): ops should not track, so backward becomes a no-op.
    w = Tensor(rng.standard_normal((4, 4), dtype=np.float32), requires_grad=True).to(device)
    with rasptorch.no_grad():
        loss = (w * w).mean()
    loss.backward()

    has_grad_w = (w.grad is not None) or (w.grad_vkbuf is not None)
    print(f"[no_grad] w.grad is None? {not has_grad_w}")


def main() -> None:
    parser = argparse.ArgumentParser(description="rasptorch essentials demo (CPU + GPU-autograd)")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Use 'gpu' to run the autograd path on Vulkan (or NumPy fallback if Vulkan unavailable).",
    )
    parser.add_argument("--steps", type=int, default=5, help="Training steps for the MLP demo")
    args = parser.parse_args()

    device = str(args.device)
    print(f"rasptorch essentials demo | device={_device_banner(device)}")

    demo_softmax_logsoftmax_backward(device=device)
    demo_layernorm_dropout_mlp_training(device=device, steps=int(args.steps))
    demo_no_grad_and_detach(device=device)


if __name__ == "__main__":
    main()
