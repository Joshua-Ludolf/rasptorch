from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from . import vulkan_backend as vk
from .tensor import Parameter, Tensor, is_grad_enabled


class Module:
    def __init__(self) -> None:
        self.training: bool = True

    def parameters(self) -> Iterable[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, prefix: str = "") -> Iterable[tuple[str, Parameter]]:
        """Yield (name, Parameter) pairs, similar to torch.nn.Module.named_parameters."""

        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Parameter):
                yield full, value
            elif isinstance(value, Module):
                yield from value.named_parameters(full)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    sub = f"{full}.{i}"
                    if isinstance(item, Parameter):
                        yield sub, item
                    elif isinstance(item, Module):
                        yield from item.named_parameters(sub)

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a CPU numpy-based state_dict (suitable for torch.save)."""

        out: dict[str, np.ndarray] = {}
        for name, p in self.named_parameters():
            out[name] = np.asarray(p.numpy(), dtype=np.float32).copy()
        return out

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None
            if p.grad_vkbuf is not None:
                vk.free(p.grad_vkbuf)
                p.grad_vkbuf = None

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

    def to(self, device: str) -> "Module":
        """Move module parameters to a device ("cpu" or "gpu")."""

        for name, value in list(self.__dict__.items()):
            if isinstance(value, Parameter):
                setattr(self, name, value.to(device))
            elif isinstance(value, Module):
                value.to(device)
            elif isinstance(value, (list, tuple)):
                new_items = []
                changed = False
                for item in value:
                    if isinstance(item, Parameter):
                        new_items.append(item.to(device))
                        changed = True
                    elif isinstance(item, Module):
                        item.to(device)
                        new_items.append(item)
                    else:
                        new_items.append(item)
                if changed:
                    setattr(self, name, type(value)(new_items))
        return self

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
        track = is_grad_enabled() and (
            x.requires_grad
            or self.weight.requires_grad
            or (self.bias is not None and self.bias.requires_grad)
        )
        # GPU path: explicit Vulkan kernels (keeps grads on GPU).
        if x.device == "gpu" or self.weight.device == "gpu" or (self.bias is not None and self.bias.device == "gpu"):
            # weight stored as [out,in] so forward uses x @ weight.T
            w_t = self.weight.T
            out = x @ w_t
            if self.bias is not None:
                out = out.add_rowvec(self.bias)

            out._op = "linear"

            if track:
                out.requires_grad = True
                out._prev = {x}  # params handled in closure

                def _backward() -> None:
                    if out.grad_vkbuf is None:
                        return
                    grad_out = out.grad_vkbuf

                    # dX = dY @ W   where W is [out,in]
                    if x.requires_grad:
                        x._accum_grad_vk(vk.matmul(grad_out, self.weight._as_vkbuf()))

                    # dW = dY^T @ X
                    if self.weight.requires_grad:
                        go_t = vk.transpose2d(grad_out)
                        self.weight._accum_grad_vk(vk.matmul(go_t, x._as_vkbuf()))
                        vk.free(go_t)

                    # dB = sum_rows(dY)
                    if self.bias is not None and self.bias.requires_grad:
                        self.bias._accum_grad_vk(vk.reduce_sum_rows(grad_out))

                out._backward = _backward
            else:
                out.requires_grad = False
            return out

        # CPU path: manual linear layer using raw arrays.
        out_data = x.data @ self.weight.data.T
        if self.bias is not None:
            out_data = out_data + self.bias.data

        out = Tensor(out_data, requires_grad=track, device=x.device)

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

        out._op = "linear"
        if track:
            out._prev = {x}  # we treat params separately in the closure
            out._backward = _backward
        return out


class Conv2d(Module):
    """A minimal 2D convolution layer (NCHW).

    Currently supports:
    - groups=1 only
    - dilation=1 only
    - stride/padding as ints or 2-tuples
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = int(kernel_size)
        else:
            kh, kw = int(kernel_size[0]), int(kernel_size[1])
        if isinstance(stride, int):
            sh = sw = int(stride)
        else:
            sh, sw = int(stride[0]), int(stride[1])
        if isinstance(padding, int):
            ph = pw = int(padding)
        else:
            ph, pw = int(padding[0]), int(padding[1])

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kh, kw)
        self.stride = (sh, sw)
        self.padding = (ph, pw)

        fan_in = self.in_channels * kh * kw
        limit = np.sqrt(2.0 / fan_in)
        w = np.random.randn(self.out_channels, self.in_channels, kh, kw).astype("float32") * limit
        self.weight = Parameter(w)
        self.bias = Parameter(np.zeros(self.out_channels, dtype="float32")) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"Conv2d expects NCHW input, got shape={x.shape}")
        N, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Conv2d in_channels mismatch: expected {self.in_channels}, got {C}")

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        OH = (H + 2 * ph - kh) // sh + 1
        OW = (W + 2 * pw - kw) // sw + 1
        if OH <= 0 or OW <= 0:
            raise ValueError("Conv2d output spatial dims must be positive")

        # GPU path
        if x.device == "gpu" or self.weight.device == "gpu" or (self.bias is not None and self.bias.device == "gpu"):
            track = is_grad_enabled() and (
                x.requires_grad
                or self.weight.requires_grad
                or (self.bias is not None and self.bias.requires_grad)
            )
            xbuf = x._as_vkbuf()
            # im2col: [N*OH*OW, C*kh*kw]
            xcol = vk.im2col_nchw(xbuf, kh=kh, kw=kw, stride_h=sh, stride_w=sw, pad_h=ph, pad_w=pw)

            # weight as [out, K] where K=C*kh*kw
            K = int(self.in_channels * kh * kw)
            w2d = vk.view(self.weight._as_vkbuf(), (self.out_channels, K))
            w2d_t = vk.transpose2d(w2d)  # [K,out]
            vk.free(w2d)

            y2d = vk.matmul(xcol, w2d_t)  # [N*OH*OW, out]
            vk.free(w2d_t)
            if self.bias is not None:
                y2d2 = vk.add_rowvec(y2d, self.bias._as_vkbuf())
                vk.free(y2d)
                y2d = y2d2

            y = vk.mat2nchw(y2d, out_shape=(int(N), int(self.out_channels), int(OH), int(OW)))
            vk.free(y2d)

            out = Tensor._from_vkbuf(y, requires_grad=track)
            out._op = "conv2d"

            if track:
                out._prev = {x}

                def _backward() -> None:
                    if out.grad_vkbuf is None:
                        return

                    dY_nchw = out.grad_vkbuf  # [N,out,OH,OW]
                    dY2d = vk.nchw2mat(dY_nchw)  # [N*OH*OW, out]

                    # Bias grad
                    if self.bias is not None and self.bias.requires_grad:
                        self.bias._accum_grad_vk(vk.reduce_sum_rows(dY2d))

                    # Weight grad: dW(out,K) = dY^T(out,R) @ Xcol(R,K)
                    if self.weight.requires_grad:
                        dY2d_t = vk.transpose2d(dY2d)  # [out, R]
                        dW2d = vk.matmul(dY2d_t, xcol)  # [out, K]
                        vk.free(dY2d_t)
                        dW4 = vk.view(dW2d, self.weight.shape)
                        self.weight._accum_grad_vk(dW4)
                        vk.free(dW2d)

                    # Input grad: dXcol(R,K) = dY(R,out) @ W(out,K)
                    if x.requires_grad:
                        w2d_local = vk.view(self.weight._as_vkbuf(), (self.out_channels, K))
                        dXcol = vk.matmul(dY2d, w2d_local)  # [R,K]
                        vk.free(w2d_local)
                        dX = vk.col2im_nchw(
                            dXcol,
                            out_shape=(int(N), int(self.in_channels), int(H), int(W)),
                            kh=kh,
                            kw=kw,
                            stride_h=sh,
                            stride_w=sw,
                            pad_h=ph,
                            pad_w=pw,
                        )
                        vk.free(dXcol)
                        x._accum_grad_vk(dX)

                    vk.free(dY2d)
                    vk.free(xcol)

                out._backward = _backward
            else:
                # In inference/no_grad cases, don't leak the saved im2col buffer.
                vk.free(xcol)
            return out

        # CPU path
        xx = x.data
        ww = self.weight.data
        bb = self.bias.data if self.bias is not None else None

        K = int(self.in_channels * kh * kw)
        # im2col
        xcol_np = np.zeros((N * OH * OW, K), dtype=np.float32)
        row = 0
        for n in range(N):
            for oh in range(OH):
                for ow in range(OW):
                    col_idx = 0
                    for c in range(C):
                        for rkh in range(kh):
                            for rkw in range(kw):
                                ih = oh * sh + rkh - ph
                                iw = ow * sw + rkw - pw
                                if 0 <= ih < H and 0 <= iw < W:
                                    xcol_np[row, col_idx] = xx[n, c, ih, iw]
                                else:
                                    xcol_np[row, col_idx] = 0.0
                                col_idx += 1
                    row += 1

        w2d_np = ww.reshape(self.out_channels, K)
        y2d_np = xcol_np @ w2d_np.T
        if bb is not None:
            y2d_np = y2d_np + bb
        y_np = y2d_np.reshape(N, OH, OW, self.out_channels).transpose(0, 3, 1, 2)

        track = is_grad_enabled() and (
            x.requires_grad
            or self.weight.requires_grad
            or (self.bias is not None and self.bias.requires_grad)
        )
        out = Tensor(y_np, requires_grad=track, device=x.device)
        out._op = "conv2d"
        if track:
            out._prev = {x}

        def _backward() -> None:
            if out.grad is None:
                return
            dY = out.grad  # [N,out,OH,OW]
            dY2d_np = dY.transpose(0, 2, 3, 1).reshape(N * OH * OW, self.out_channels)

            if self.bias is not None and self.bias.requires_grad:
                gb = dY2d_np.sum(axis=0)
                self.bias.grad = gb if self.bias.grad is None else (self.bias.grad + gb)

            if self.weight.requires_grad:
                dW2d_np = dY2d_np.T @ xcol_np  # [out,K]
                dW_np = dW2d_np.reshape(self.weight.data.shape)
                self.weight.grad = dW_np if self.weight.grad is None else (self.weight.grad + dW_np)

            if x.requires_grad:
                dXcol_np = dY2d_np @ w2d_np  # [R,K]
                dX_np = np.zeros((N, C, H, W), dtype=np.float32)
                row = 0
                for n in range(N):
                    for oh in range(OH):
                        for ow in range(OW):
                            col_idx = 0
                            for c in range(C):
                                for rkh in range(kh):
                                    for rkw in range(kw):
                                        ih = oh * sh + rkh - ph
                                        iw = ow * sw + rkw - pw
                                        if 0 <= ih < H and 0 <= iw < W:
                                            dX_np[n, c, ih, iw] += dXcol_np[row, col_idx]
                                        col_idx += 1
                            row += 1
                x.grad = dX_np if x.grad is None else (x.grad + dX_np)

        if track:
            out._backward = _backward
        return out


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if len(x.shape) < 2:
            return x
        n = x.shape[0]
        rest = int(np.prod(x.shape[1:]))
        return x.view(n, rest)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x.tanh()


class Dropout(Module):
    """Dropout layer.

    During training:
      out = x * mask / (1-p)
    During eval:
      out = x
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        p = float(p)
        if p < 0.0 or p >= 1.0:
            raise ValueError("Dropout p must be in [0, 1)")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - float(self.p)
        mask = (np.random.random_sample(x.shape) < keep).astype(np.float32) / max(keep, 1e-12)
        m = Tensor(mask).to(x.device)
        return x * m


class LayerNorm(Module):
    """LayerNorm over the last N dimensions.

    This matches the common PyTorch usage: normalize over the last dimensions
    specified by normalized_shape.
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            norm_shape = (int(normalized_shape),)
        else:
            norm_shape = tuple(int(s) for s in normalized_shape)
        if any(s <= 0 for s in norm_shape):
            raise ValueError("normalized_shape entries must be > 0")
        self.normalized_shape = norm_shape
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)

        if self.elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32))
            self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if x.device == "gpu":
            if len(self.normalized_shape) != 1:
                raise NotImplementedError("LayerNorm GPU path currently supports 1D normalized_shape only")
            if len(x.shape) != 2:
                raise NotImplementedError("LayerNorm GPU path currently supports 2D inputs only")
            if tuple(x.shape[-1:]) != tuple(self.normalized_shape):
                raise ValueError(
                    f"LayerNorm normalized_shape mismatch: expected trailing {self.normalized_shape}, got {tuple(x.shape)}"
                )
            if abs(self.eps - 1e-5) > 1e-12:
                raise NotImplementedError("LayerNorm GPU path currently supports eps=1e-5 only")

            track = is_grad_enabled() and (
                x.requires_grad
                or (self.weight is not None and self.weight.requires_grad)
                or (self.bias is not None and self.bias.requires_grad)
            )

            xhat_buf = vk.layernorm2d(x._as_vkbuf())
            xhat_track = is_grad_enabled() and x.requires_grad
            xhat = Tensor._from_vkbuf(xhat_buf, requires_grad=xhat_track)
            xhat._op = "layer_norm"

            if xhat_track:
                xhat._prev = {x}

                def _backward() -> None:
                    if xhat.grad_vkbuf is None:
                        return
                    dx = vk.layernorm2d_backward(x._as_vkbuf(), xhat.grad_vkbuf)
                    x._accum_grad_vk(dx)

                xhat._backward = _backward

            out = xhat
            if self.weight is not None:
                out = out.mul_rowvec(self.weight)
            if self.bias is not None:
                out = out.add_rowvec(self.bias)
            out.requires_grad = track
            return out

        # CPU path

        if len(x.shape) < len(self.normalized_shape):
            raise ValueError(f"LayerNorm input rank {len(x.shape)} < normalized_shape rank {len(self.normalized_shape)}")
        if tuple(x.shape[-len(self.normalized_shape) :]) != tuple(self.normalized_shape):
            raise ValueError(
                f"LayerNorm normalized_shape mismatch: expected trailing {self.normalized_shape}, got {tuple(x.shape)}"
            )

        track = is_grad_enabled() and (
            x.requires_grad
            or (self.weight is not None and self.weight.requires_grad)
            or (self.bias is not None and self.bias.requires_grad)
        )

        x_np = x.numpy()
        axes = tuple(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))
        mean = x_np.mean(axis=axes, keepdims=True)
        var = ((x_np - mean) ** 2).mean(axis=axes, keepdims=True)
        invstd = 1.0 / np.sqrt(var + self.eps)
        xhat = (x_np - mean) * invstd

        if self.weight is not None:
            w = self.weight.numpy().reshape((1,) * (xhat.ndim - len(self.normalized_shape)) + self.normalized_shape)
            y = xhat * w
        else:
            w = None
            y = xhat

        if self.bias is not None:
            b = self.bias.numpy().reshape((1,) * (xhat.ndim - len(self.normalized_shape)) + self.normalized_shape)
            y = y + b

        out = Tensor(y.astype(np.float32, copy=False), requires_grad=track, device="cpu")
        out._op = "layer_norm"
        if not track:
            return out

        out._prev = {x}

        # Precompute for backward
        N = float(np.prod(self.normalized_shape))

        def _backward() -> None:
            if out.grad is None:
                return
            dy = out.grad

            # grads for affine
            dxhat = dy
            if self.weight is not None:
                w_b = w  # broadcasted
                if self.weight.requires_grad:
                    dgamma = (dy * xhat).sum(axis=axes, keepdims=False)
                    self.weight.grad = dgamma if self.weight.grad is None else (self.weight.grad + dgamma)
                dxhat = dy * w_b
            if self.bias is not None and self.bias.requires_grad:
                dbeta = dy.sum(axis=axes, keepdims=False)
                self.bias.grad = dbeta if self.bias.grad is None else (self.bias.grad + dbeta)

            if x.requires_grad:
                # dx = (1/N)*invstd*(N*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
                sum1 = dxhat.sum(axis=axes, keepdims=True)
                sum2 = (dxhat * xhat).sum(axis=axes, keepdims=True)
                dx = (invstd / max(N, 1.0)) * (dxhat * N - sum1 - xhat * sum2)
                x.grad = dx if x.grad is None else (x.grad + dx)

        out._backward = _backward
        return out


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers: List[Module] = list(layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        for layer in self.layers:
            x = layer(x)
        return x
