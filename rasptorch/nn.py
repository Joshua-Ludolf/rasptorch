from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .tensor import Parameter, Tensor


class Module:
    def __init__(self) -> None:
        self.training: bool = True

    def parameters(self) -> Iterable[Parameter]:
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        yield from item.parameters()
                    elif isinstance(item, Parameter):
                        yield item

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        """Set the module in training or eval mode (recursively)."""
        self.training = mode
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self) -> "Module":
        """Set the module to evaluation mode."""
        return self.train(False)

    def __call__(self, *args, **kwargs):  # type: ignore[override]
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - interface only
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        limit = np.sqrt(2.0 / in_features)
        weight_data = np.random.randn(out_features, in_features).astype("float32") * limit
        self.weight = Parameter(weight_data)
        if bias:
            self.bias = Parameter(np.zeros(out_features, dtype="float32"))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Manual linear layer using raw arrays, with custom backward that
        # handles bias gradients correctly (no broadcast shape issues).
        out_data = x.data @ self.weight.data.T
        if self.bias is not None:
            out_data = out_data + self.bias.data

        requires_grad = (
            x.requires_grad or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad)
        )
        out = Tensor(out_data, requires_grad=requires_grad, device=x.device)

        def _backward() -> None:
            if out.grad is None:
                return
            grad_out = out.grad

            # dL/dx = dL/dy @ W
            if x.requires_grad:
                grad_x = grad_out @ self.weight.data
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad = x.grad + grad_x

            # dL/dW = grad_out^T @ x
            if self.weight.requires_grad:
                grad_w = grad_out.T @ x.data
                if self.weight.grad is None:
                    self.weight.grad = grad_w
                else:
                    self.weight.grad = self.weight.grad + grad_w

            # dL/db = sum over batch
            if self.bias is not None and self.bias.requires_grad:
                grad_b = grad_out.sum(axis=0)
                if self.bias.grad is None:
                    self.bias.grad = grad_b
                else:
                    self.bias.grad = self.bias.grad + grad_b

        out._backward = _backward
        out._op = "linear"
        out._prev = {x}  # we treat params separately in the closure
        return out


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x.tanh()


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers: List[Module] = list(layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        for layer in self.layers:
            x = layer(x)
        return x
