from __future__ import annotations

from contextlib import contextmanager

from . import vulkan_backend as vk


_compute_dtype: str = "float32"


def get_compute_dtype() -> str:
    return _compute_dtype


@contextmanager
def autocast(dtype: str = "float16"):
    global _compute_dtype
    prev = _compute_dtype
    _compute_dtype = str(dtype)
    try:
        yield
    finally:
        _compute_dtype = prev


class GradScaler:
    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        self._scale = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self.enabled = bool(enabled)
        self._growth_tracker = 0

    def get_scale(self) -> float:
        return float(self._scale)

    def scale(self, loss):
        if not self.enabled:
            return loss
        return loss * self._scale

    def unscale_(self, optimizer) -> None:
        if not self.enabled:
            return
        inv = 1.0 / max(self._scale, 1e-20)
        for p in getattr(optimizer, "_params", []):
            if p.grad_vkbuf is not None:
                scaled = vk.mul_scalar(p.grad_vkbuf, inv)
                vk.free(p.grad_vkbuf)
                p.grad_vkbuf = scaled
            elif p.grad is not None:
                p.grad = p.grad * inv

    def step(self, optimizer) -> None:
        self.unscale_(optimizer)
        optimizer.step()

    def update(self, found_inf: bool = False) -> None:
        if not self.enabled:
            return
        if found_inf:
            self._scale = max(1.0, self._scale * self.backoff_factor)
            self._growth_tracker = 0
            return
        self._growth_tracker += 1
        if self._growth_tracker >= self.growth_interval:
            self._scale *= self.growth_factor
            self._growth_tracker = 0