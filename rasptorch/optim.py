from __future__ import annotations

from typing import Iterable

import numpy as np

from . import vulkan_backend as vk
from .tensor import Parameter


def _param_arrays(param: Parameter) -> tuple[np.ndarray, np.ndarray] | None:
    if param.device == "gpu":
        if param.grad_vkbuf is None or param._vkbuf is None:
            return None
        return vk.to_cpu(param._vkbuf).astype(np.float32), vk.to_cpu(param.grad_vkbuf).astype(np.float32)
    if param.grad is None:
        return None
    return param.data.astype(np.float32, copy=True), param.grad.astype(np.float32, copy=True)


def _write_param_array(param: Parameter, value: np.ndarray) -> None:
    arr = np.asarray(value, dtype=np.float32)
    if param.device == "gpu":
        if param._vkbuf is None:
            raise RuntimeError("GPU parameter is missing its Vulkan buffer")
        vk.write(param._vkbuf, arr)
    else:
        param.data = arr


def _ensure_gpu_state(
    store: dict[int, vk.VulkanBuffer],
    key: int,
    ref: vk.VulkanBuffer,
) -> vk.VulkanBuffer:
    state = store.get(key)
    if state is None or state.shape != ref.shape:
        if state is not None:
            vk.free(state)
        state = vk.zeros_like(ref)
        store[key] = state
    return state


def _ensure_cpu_state(store: dict[int, np.ndarray], key: int, shape: tuple[int, ...]) -> np.ndarray:
    state = store.get(key)
    if state is None or state.shape != shape:
        state = np.zeros(shape, dtype=np.float32)
        store[key] = state
    return state


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


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._step = 0
        self._m_cpu: dict[int, np.ndarray] = {}
        self._v_cpu: dict[int, np.ndarray] = {}
        self._m_gpu: dict[int, vk.VulkanBuffer] = {}
        self._v_gpu: dict[int, vk.VulkanBuffer] = {}

    def step(self) -> None:
        self._step += 1
        beta1, beta2 = self.betas

        for p in self._params:
            if p.device == "gpu":
                if p.grad_vkbuf is None or p._vkbuf is None:
                    continue
                key = id(p)
                m_state = _ensure_gpu_state(self._m_gpu, key, p._vkbuf)
                v_state = _ensure_gpu_state(self._v_gpu, key, p._vkbuf)
                vk.adam_update_inplace(
                    p._vkbuf,
                    p.grad_vkbuf,
                    m_state,
                    v_state,
                    lr=self.lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=self.eps,
                    bias_correction1=1.0 - beta1 ** self._step,
                    bias_correction2=1.0 - beta2 ** self._step,
                    weight_decay=self.weight_decay,
                )
                continue

            payload = _param_arrays(p)
            if payload is None:
                continue
            param_np, grad_np = payload
            key = id(p)

            m_np = _ensure_cpu_state(self._m_cpu, key, p.data.shape)
            v_np = _ensure_cpu_state(self._v_cpu, key, p.data.shape)

            if self.weight_decay != 0.0:
                grad_np = grad_np + self.weight_decay * param_np

            m_np[...] = beta1 * m_np + (1.0 - beta1) * grad_np
            v_np[...] = beta2 * v_np + (1.0 - beta2) * (grad_np ** 2)

            m_hat = m_np / (1.0 - beta1 ** self._step)
            v_hat = v_np / (1.0 - beta2 ** self._step)
            param_np = param_np - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))

            _write_param_array(p, param_np)


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params)
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._step = 0
        self._m_cpu: dict[int, np.ndarray] = {}
        self._v_cpu: dict[int, np.ndarray] = {}
        self._m_gpu: dict[int, vk.VulkanBuffer] = {}
        self._v_gpu: dict[int, vk.VulkanBuffer] = {}

    def step(self) -> None:
        self._step += 1
        beta1, beta2 = self.betas

        for p in self._params:
            if p.device == "gpu":
                if p.grad_vkbuf is None or p._vkbuf is None:
                    continue
                key = id(p)
                m_state = _ensure_gpu_state(self._m_gpu, key, p._vkbuf)
                v_state = _ensure_gpu_state(self._v_gpu, key, p._vkbuf)
                vk.adamw_update_inplace(
                    p._vkbuf,
                    p.grad_vkbuf,
                    m_state,
                    v_state,
                    lr=self.lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=self.eps,
                    bias_correction1=1.0 - beta1 ** self._step,
                    bias_correction2=1.0 - beta2 ** self._step,
                    weight_decay=self.weight_decay,
                )
                continue

            payload = _param_arrays(p)
            if payload is None:
                continue
            param_np, grad_np = payload
            key = id(p)

            m_np = _ensure_cpu_state(self._m_cpu, key, p.data.shape)
            v_np = _ensure_cpu_state(self._v_cpu, key, p.data.shape)

            param_np = param_np * (1.0 - self.lr * self.weight_decay)
            m_np[...] = beta1 * m_np + (1.0 - beta1) * grad_np
            v_np[...] = beta2 * v_np + (1.0 - beta2) * (grad_np ** 2)

            m_hat = m_np / (1.0 - beta1 ** self._step)
            v_hat = v_np / (1.0 - beta2 ** self._step)
            param_np = param_np - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))

            _write_param_array(p, param_np)


class RMSProp(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-2,
        rho: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = float(lr)
        self.rho = float(rho)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._v_cpu: dict[int, np.ndarray] = {}
        self._v_gpu: dict[int, vk.VulkanBuffer] = {}

    def step(self) -> None:
        for p in self._params:
            if p.device == "gpu":
                if p.grad_vkbuf is None or p._vkbuf is None:
                    continue
                key = id(p)
                v_state = _ensure_gpu_state(self._v_gpu, key, p._vkbuf)
                vk.rmsprop_update_inplace(
                    p._vkbuf,
                    p.grad_vkbuf,
                    v_state,
                    lr=self.lr,
                    rho=self.rho,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
                continue

            payload = _param_arrays(p)
            if payload is None:
                continue
            param_np, grad_np = payload
            key = id(p)

            v_np = _ensure_cpu_state(self._v_cpu, key, p.data.shape)

            if self.weight_decay != 0.0:
                grad_np = grad_np + self.weight_decay * param_np

            v_np[...] = self.rho * v_np + (1.0 - self.rho) * (grad_np ** 2)
            param_np = param_np - self.lr * grad_np / (np.sqrt(v_np) + self.eps)

            _write_param_array(p, param_np)
