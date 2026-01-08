from __future__ import annotations

from typing import Iterable

from .tensor import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter]) -> None:
        self._params = list(params)

    def step(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params: Iterable[Parameter], lr: float = 1e-3) -> None:
        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        for p in self._params:
            if p.grad is None:
                continue
            p.data = p.data - self.lr * p.grad
