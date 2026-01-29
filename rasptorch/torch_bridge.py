from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .tensor import Tensor
from . import vulkan_backend as vk


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for rasptorch.torch_bridge. Install torch (PyTorch 2.2–2.3 recommended on Pi)."
        ) from e


def to_rasptorch(x, *, device: str = "gpu") -> Tensor:
    """Convert a torch.Tensor to a rasptorch.Tensor.

    Notes:
    - This is a **bridge**: we materialize a CPU numpy array first.
    - Only float32 is supported in the bridge for now.
    """

    torch = _require_torch()
    if not isinstance(x, torch.Tensor):
        raise TypeError("to_rasptorch expects a torch.Tensor")

    x_cpu = x.detach().to(device="cpu")
    if x_cpu.dtype != torch.float32:
        x_cpu = x_cpu.float()
    x_np = x_cpu.contiguous().numpy()
    return Tensor(np.asarray(x_np, dtype=np.float32)).to(device)


def to_torch(x: Tensor):
    """Convert a rasptorch.Tensor to a torch.Tensor (CPU)."""

    torch = _require_torch()
    return torch.from_numpy(np.asarray(x.numpy(), dtype=np.float32))


@dataclass
class RaspLinear:
    """torch.nn.Module-like Linear backed by rasptorch Vulkan kernels.

    This is implemented as a small callable object so we don't depend on torch at import time.
    """

    weight: "object"  # torch.Tensor [out,in]
    bias: Optional["object"]  # torch.Tensor [out]

    @classmethod
    def from_torch(cls, mod) -> "RaspLinear":
        torch = _require_torch()
        if not isinstance(mod, torch.nn.Linear):
            raise TypeError("expected torch.nn.Linear")
        w = mod.weight.detach().to(dtype=torch.float32, device="cpu").contiguous()
        b = None
        if mod.bias is not None:
            b = mod.bias.detach().to(dtype=torch.float32, device="cpu").contiguous()
        return cls(weight=w, bias=b)

    def __call__(self, x):
        torch = _require_torch()
        if not isinstance(x, torch.Tensor):
            raise TypeError("RaspLinear expects torch.Tensor input")
        # Fallback: only 2D [N, in] supported.
        if x.ndim != 2:
            raise ValueError(f"RaspLinear expects 2D input [N,in], got shape={tuple(x.shape)}")

        # Convert inputs + params to rasptorch GPU
        rx = to_rasptorch(x, device="gpu")
        rw = Tensor(self.weight.numpy()).to("gpu")
        # weight is [out,in] so use x @ w.T
        y = rx @ rw.T
        if self.bias is not None:
            rb = Tensor(self.bias.numpy()).to("gpu")
            y = y.add_rowvec(rb)
        return to_torch(y)


@dataclass
class RaspConv2d:
    """torch.nn.Module-like Conv2d backed by rasptorch Vulkan kernels (inference).

    Current limitations:
    - groups=1 only
    - dilation=1 only
    - stride/padding as ints or 2-tuples
    - float32 only
    """

    weight: "object"  # torch.Tensor [out,in,kh,kw]
    bias: Optional["object"]
    stride: Tuple[int, int]
    padding: Tuple[int, int]

    @classmethod
    def from_torch(cls, mod) -> "RaspConv2d":
        torch = _require_torch()
        if not isinstance(mod, torch.nn.Conv2d):
            raise TypeError("expected torch.nn.Conv2d")
        if mod.groups != 1 or mod.dilation != (1, 1):
            raise ValueError("RaspConv2d currently supports groups=1 and dilation=1 only")

        w = mod.weight.detach().to(dtype=torch.float32, device="cpu").contiguous()
        b = None
        if mod.bias is not None:
            b = mod.bias.detach().to(dtype=torch.float32, device="cpu").contiguous()

        sh, sw = (int(mod.stride[0]), int(mod.stride[1]))
        ph, pw = (int(mod.padding[0]), int(mod.padding[1]))
        return cls(weight=w, bias=b, stride=(sh, sw), padding=(ph, pw))

    def __call__(self, x):
        torch = _require_torch()
        if not isinstance(x, torch.Tensor):
            raise TypeError("RaspConv2d expects torch.Tensor input")
        if x.ndim != 4:
            raise ValueError(f"RaspConv2d expects NCHW input, got shape={tuple(x.shape)}")

        # Convert x to rasptorch GPU
        rx = to_rasptorch(x, device="gpu")
        xbuf = rx._as_vkbuf()

        # Upload weights/bias to GPU
        w_np = self.weight.numpy()
        wbuf = vk.to_gpu(np.asarray(w_np, dtype=np.float32))
        try:
            if self.bias is not None:
                bbuf = vk.to_gpu(np.asarray(self.bias.numpy(), dtype=np.float32))
            else:
                bbuf = None

            kh, kw = int(w_np.shape[2]), int(w_np.shape[3])
            sh, sw = self.stride
            ph, pw = self.padding

            # xcol: [N*OH*OW, C*kh*kw]
            xcol = vk.im2col_nchw(xbuf, kh=kh, kw=kw, stride_h=sh, stride_w=sw, pad_h=ph, pad_w=pw)
            try:
                out_ch = int(w_np.shape[0])
                in_ch = int(w_np.shape[1])
                K = int(in_ch * kh * kw)

                w2d = vk.view(wbuf, (out_ch, K))
                try:
                    w2d_t = vk.transpose2d(w2d)  # [K,out]
                finally:
                    vk.free(w2d)

                try:
                    y2d = vk.matmul(xcol, w2d_t)  # [N*OH*OW, out]
                finally:
                    vk.free(w2d_t)

                if bbuf is not None:
                    y2d2 = vk.add_rowvec(y2d, bbuf)
                    vk.free(y2d)
                    y2d = y2d2

                # Infer output spatial dims
                N, C, H, W = x.shape
                OH = (int(H) + 2 * ph - kh) // sh + 1
                OW = (int(W) + 2 * pw - kw) // sw + 1

                y = vk.mat2nchw(y2d, out_shape=(int(N), int(out_ch), int(OH), int(OW)))
                vk.free(y2d)

                out = Tensor._from_vkbuf(y)
                return to_torch(out)
            finally:
                vk.free(xcol)
        finally:
            vk.free(wbuf)
            if bbuf is not None:
                vk.free(bbuf)


def convert_torch_model(model, *, device: str = "gpu"):
    """Return a shallow-converted model callable with supported layers replaced.

    This is intentionally minimal: it supports Sequential-like graphs and common modules.
    Unsupported modules are left as-is (CPU fallback).
    """

    torch = _require_torch()

    if isinstance(model, torch.nn.Sequential):
        layers = []
        for m in model:
            layers.append(convert_torch_model(m, device=device))
        return torch.nn.Sequential(*layers)

    if isinstance(model, torch.nn.Conv2d):
        # Wrap with a torch.nn.Module so it composes naturally
        class _Wrap(torch.nn.Module):
            def __init__(self, mod):
                super().__init__()
                self._rasp = RaspConv2d.from_torch(mod)

            def forward(self, x):
                # Ensure Vulkan is initialized; strict mode for GPU path.
                if device == "gpu":
                    vk.init(strict=True)
                return self._rasp(x)

        return _Wrap(model)

    if isinstance(model, torch.nn.Linear):
        class _Wrap(torch.nn.Module):
            def __init__(self, mod):
                super().__init__()
                self._rasp = RaspLinear.from_torch(mod)

            def forward(self, x):
                if device == "gpu":
                    vk.init(strict=True)
                return self._rasp(x)

        return _Wrap(model)

    if isinstance(model, torch.nn.ReLU):
        class _Wrap(torch.nn.Module):
            def forward(self, x):
                # Fast path: use Vulkan relu by round-tripping through rasptorch.
                if device == "gpu":
                    vk.init(strict=True)
                    return to_torch(to_rasptorch(x, device="gpu").relu())
                return torch.relu(x)

        return _Wrap()

    # Fallback: keep module unchanged
    return model
