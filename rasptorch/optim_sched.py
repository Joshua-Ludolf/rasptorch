from __future__ import annotations

import math
from collections.abc import Iterable


class _LRScheduler:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.base_lr = float(optimizer.lr)
        self.last_epoch = -1

    def get_last_lr(self) -> list[float]:
        return [float(self.optimizer.lr)]


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(optimizer)
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def step(self) -> None:
        self.last_epoch += 1
        completed_steps = (self.last_epoch + 1) // self.step_size
        self.optimizer.lr = self.base_lr * (self.gamma ** completed_steps)


class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones: Iterable[int], gamma: float = 0.1) -> None:
        super().__init__(optimizer)
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = float(gamma)

    def step(self) -> None:
        self.last_epoch += 1
        count = sum(1 for m in self.milestones if self.last_epoch >= m)
        self.optimizer.lr = self.base_lr * (self.gamma ** count)


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma: float) -> None:
        super().__init__(optimizer)
        self.gamma = float(gamma)

    def step(self) -> None:
        self.last_epoch += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** (self.last_epoch + 1))


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0) -> None:
        super().__init__(optimizer)
        if T_max <= 0:
            raise ValueError("T_max must be > 0")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)

    def step(self) -> None:
        self.last_epoch += 1
        progress = min(self.last_epoch, self.T_max)
        cosine = (1.0 + math.cos(math.pi * progress / self.T_max)) / 2.0
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * cosine


class ReduceLROnPlateau(_LRScheduler):
    def __init__(
        self,
        optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
    ) -> None:
        super().__init__(optimizer)
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if factor <= 0.0 or factor >= 1.0:
            raise ValueError("factor must be in (0, 1)")
        if patience < 0:
            raise ValueError("patience must be >= 0")
        self.mode = mode
        self.factor = float(factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, metric: float) -> None:
        value = float(metric)
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "min":
            improved = value < (self.best - self.threshold)
        else:
            improved = value > (self.best + self.threshold)

        if improved:
            self.best = value
            self.bad_epochs = 0
            return

        self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            self.optimizer.lr = float(self.optimizer.lr) * self.factor
            self.bad_epochs = 0


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, after_scheduler=None) -> None:
        super().__init__(optimizer)
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be > 0")
        self.warmup_steps = int(warmup_steps)
        self.after_scheduler = after_scheduler
        self._after_started = False

    def step(self, metric: float | None = None) -> None:
        self.last_epoch += 1
        if self.last_epoch < self.warmup_steps:
            self.optimizer.lr = self.base_lr * float(self.last_epoch + 1) / float(self.warmup_steps)
            return

        if self.after_scheduler is None:
            self.optimizer.lr = self.base_lr
            return

        if not self._after_started:
            self.after_scheduler.base_lr = self.base_lr
            self.after_scheduler.last_epoch = -1
            self._after_started = True
            self.optimizer.lr = self.base_lr
            return

        if isinstance(self.after_scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError("WarmupScheduler with ReduceLROnPlateau requires metric")
            self.after_scheduler.step(metric)
        else:
            self.after_scheduler.step()