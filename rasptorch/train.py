from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

import numpy as np

from .nn import Module
from .optim import Optimizer
from .tensor import Tensor, no_grad


Array = np.ndarray


class Metric(Protocol):
    name: str

    def reset(self) -> None: ...

    def update(self, logits: Tensor, y: Array) -> None: ...

    def compute(self) -> float: ...


class Accuracy:
    """Top-1 accuracy for classification.

    Expects:
    - logits: Tensor of shape [N, C]
    - y: integer labels of shape [N]

    Notes:
    - Uses `logits.numpy()` under the hood (GPU readback if needed).
    """

    name = "acc"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, logits: Tensor, y: Array) -> None:
        y_int = np.asarray(y).reshape(-1).astype(np.int64, copy=False)
        pred = logits.numpy().reshape((y_int.size, -1)).argmax(axis=1)
        self.correct += int((pred == y_int).sum())
        self.total += int(y_int.size)

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return float(self.correct / self.total)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self.sum / self.count)


@dataclass
class EpochStats:
    loss: float
    metrics: dict[str, float]
    seconds: float
    samples_per_sec: float


TargetTransform = Callable[[Any], Tensor]


def classification_target_one_hot(num_classes: int, *, device: str = "cpu") -> TargetTransform:
    """Create a target transform that converts integer labels -> one-hot Tensor."""

    def _xf(y: Any) -> Tensor:
        y_int = np.asarray(y).reshape(-1).astype(np.int64)
        out = np.zeros((int(y_int.size), int(num_classes)), dtype=np.float32)
        out[np.arange(y_int.size), y_int] = 1.0
        return Tensor(out).to(device)

    return _xf


def _to_device_tensor(x: Any, device: str) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device)
    arr = np.asarray(x)
    # default float32
    return Tensor(arr.astype(np.float32, copy=False)).to(device)


def train_one_epoch(
    model: Module,
    optimizer: Optimizer,
    loader: Iterable,
    *,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: str,
    target_transform: Optional[TargetTransform] = None,
    metrics: Optional[Sequence[Metric]] = None,
    log_every: int = 0,
) -> EpochStats:
    model.train(True)

    loss_meter = AverageMeter()
    metric_objs = list(metrics or [])
    for m in metric_objs:
        m.reset()

    t0 = time.perf_counter()
    seen = 0
    step = 0

    for batch in loader:
        step += 1
        if isinstance(batch, tuple) and len(batch) == 2:
            xb, yb = batch
        else:
            raise ValueError("Expected loader to yield (x, y) batches")

        x = _to_device_tensor(xb, device)
        if target_transform is None:
            y_t = _to_device_tensor(yb, device)
        else:
            y_t = target_transform(yb)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()

        bs = int(np.asarray(yb).shape[0])
        loss_val = float(loss.numpy().reshape(-1)[0])
        loss_meter.update(loss_val, bs)
        for m in metric_objs:
            m.update(logits, np.asarray(yb))

        seen += bs

        if log_every and (step % log_every == 0):
            # Lightweight step log (still may trigger GPU readback for loss)
            metric_str = " ".join(f"{m.name}={m.compute()*100:.2f}%" for m in metric_objs)
            if metric_str:
                metric_str = " " + metric_str
            print(f"  step {step:05d} loss={loss_meter.avg():.4f}{metric_str}")

    dt = time.perf_counter() - t0
    ips = seen / max(1e-9, dt)

    return EpochStats(
        loss=loss_meter.avg(),
        metrics={m.name: m.compute() for m in metric_objs},
        seconds=float(dt),
        samples_per_sec=float(ips),
    )


def evaluate(
    model: Module,
    loader: Iterable,
    *,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: str,
    target_transform: Optional[TargetTransform] = None,
    metrics: Optional[Sequence[Metric]] = None,
) -> EpochStats:
    model.eval()

    loss_meter = AverageMeter()
    metric_objs = list(metrics or [])
    for m in metric_objs:
        m.reset()

    t0 = time.perf_counter()
    seen = 0

    with no_grad():
        for batch in loader:
            if isinstance(batch, tuple) and len(batch) == 2:
                xb, yb = batch
            else:
                raise ValueError("Expected loader to yield (x, y) batches")

            x = _to_device_tensor(xb, device)
            if target_transform is None:
                y_t = _to_device_tensor(yb, device)
            else:
                y_t = target_transform(yb)

            logits = model(x)
            loss = loss_fn(logits, y_t)

            bs = int(np.asarray(yb).shape[0])
            loss_val = float(loss.numpy().reshape(-1)[0])
            loss_meter.update(loss_val, bs)
            for m in metric_objs:
                m.update(logits, np.asarray(yb))

            seen += bs

    dt = time.perf_counter() - t0
    ips = seen / max(1e-9, dt)

    return EpochStats(
        loss=loss_meter.avg(),
        metrics={m.name: m.compute() for m in metric_objs},
        seconds=float(dt),
        samples_per_sec=float(ips),
    )


def fit(
    model: Module,
    optimizer: Optimizer,
    train_loader: Iterable,
    *,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: str = "cpu",
    epochs: int = 10,
    val_loader: Optional[Iterable] = None,
    target_transform: Optional[TargetTransform] = None,
    metrics: Optional[Sequence[Metric]] = None,
    log_every: int = 0,
) -> None:
    """Generic training loop with PyTorch-like logging.

    Typical usage:
        metrics = [train.Accuracy()]
        tgt = train.classification_target_one_hot(10, device='gpu')
        train.fit(model, opt, train_loader, loss_fn=F.cross_entropy, device='gpu', val_loader=val_loader,
                  target_transform=tgt, metrics=metrics)

    Notes:
    - Metrics that depend on logits call `.numpy()` and may cause GPU readbacks.
    - `evaluate(...)` runs under `no_grad()`.
    """

    if epochs <= 0:
        raise ValueError("epochs must be > 0")

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(
            model,
            optimizer,
            train_loader,
            loss_fn=loss_fn,
            device=device,
            target_transform=target_transform,
            metrics=metrics,
            log_every=log_every,
        )

        if val_loader is not None:
            va = evaluate(
                model,
                val_loader,
                loss_fn=loss_fn,
                device=device,
                target_transform=target_transform,
                metrics=metrics,
            )
        else:
            va = None

        # Short, non-wrapping log format (two lines when val present)
        metric_str = " ".join(
            f"{k} {v*100:.2f}%" if k in {"acc", "val_acc"} else f"{k} {v:.4f}" for k, v in tr.metrics.items()
        )
        if metric_str:
            metric_str = " | " + metric_str

        print(
            f"Epoch {epoch:03d}/{epochs:03d}"
            f" | train loss {tr.loss:.4f}{metric_str}"
            f" | {tr.samples_per_sec:.1f} samp/s ({tr.seconds:.2f}s)"
        )

        if va is not None:
            val_metric_str = " ".join(
                f"{k} {v*100:.2f}%" if k in {"acc", "val_acc"} else f"{k} {v:.4f}" for k, v in va.metrics.items()
            )
            if val_metric_str:
                val_metric_str = " | " + val_metric_str
            print(f"           val   loss {va.loss:.4f}{val_metric_str}")
