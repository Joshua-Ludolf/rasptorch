from __future__ import annotations

from typing import Iterable

import numpy as np

from . import vulkan_backend as vk
from .tensor import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter]) -> None:
        self._params = list(params)

    def step(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None
            if p.grad_vkbuf is not None:
                vk.free(p.grad_vkbuf)
                p.grad_vkbuf = None


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)

        self._vel_cpu: dict[int, np.ndarray] = {}
        self._vel_gpu: dict[int, vk.VulkanBuffer] = {}

    def step(self) -> None:
        for p in self._params:
            if p.device == "gpu":
                if p.grad_vkbuf is None or p._vkbuf is None:
                    continue
                if self.momentum == 0.0 and self.weight_decay == 0.0:
                    vk.sgd_update_inplace(p._vkbuf, p.grad_vkbuf, self.lr)
                    continue

                key = id(p)
                v = self._vel_gpu.get(key)
                if v is None or v.shape != p._vkbuf.shape:
                    if v is not None:
                        vk.free(v)
                    v = vk.to_gpu(np.zeros(p._vkbuf.shape, dtype=np.float32))
                    self._vel_gpu[key] = v

                vk.sgd_momentum_update_inplace(
                    p._vkbuf,
                    p.grad_vkbuf,
                    v,
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )
                continue

            if p.grad is None:
                continue
            if self.momentum == 0.0 and self.weight_decay == 0.0:
                p.data = p.data - self.lr * p.grad
                continue

            key = id(p)
            v_cpu = self._vel_cpu.get(key)
            if v_cpu is None or v_cpu.shape != p.data.shape:
                v_cpu = np.zeros_like(p.data, dtype=np.float32)
                self._vel_cpu[key] = v_cpu

            g = p.grad.astype(np.float32)
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            v_cpu[...] = self.momentum * v_cpu + g
            p.data = p.data - self.lr * v_cpu
