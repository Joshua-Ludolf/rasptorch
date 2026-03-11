from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from . import init as init_ops
from . import vulkan_backend as vk
from .tensor import Parameter, Tensor, is_grad_enabled


def _make_tensor_from_np(value: np.ndarray, *, device: str, requires_grad: bool, op: str) -> Tensor:
    arr = np.asarray(value, dtype=np.float32)
    if device == "gpu":
        out = Tensor._from_vkbuf(vk.to_gpu(arr), requires_grad=requires_grad)
        out._op = op
        return out
    return Tensor(arr, requires_grad=requires_grad, device="cpu", _op=op)


def _grad_from_output(out: Tensor) -> np.ndarray | None:
    if out.device == "gpu":
        if out.grad_vkbuf is None:
            return None
        return np.asarray(vk.to_cpu(out.grad_vkbuf), dtype=np.float32)
    if out.grad is None:
        return None
    return np.asarray(out.grad, dtype=np.float32)


def _accum_tensor_grad(tensor: Tensor, grad: np.ndarray) -> None:
    if not tensor.requires_grad:
        return
    grad_np = np.asarray(grad, dtype=np.float32)
    if tensor.device == "gpu":
        tensor._accum_grad_vk(vk.to_gpu(grad_np))
    else:
        tensor.grad = grad_np if tensor.grad is None else (tensor.grad + grad_np)


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError("expected int or length-2 sequence")
        return int(value[0]), int(value[1])
    v = int(value)
    return v, v


def _batchnorm_forward(module: Module, x: Tensor, reduce_axes: tuple[int, ...], op_name: str) -> Tensor:
    x_np = np.asarray(x.numpy(), dtype=np.float32)
    stat_shape = [1] * x_np.ndim
    stat_shape[1] = x_np.shape[1]

    if module.training:
        mean = x_np.mean(axis=reduce_axes, keepdims=True)
        var = ((x_np - mean) ** 2).mean(axis=reduce_axes, keepdims=True)
        if getattr(module, "track_running_stats", True):
            momentum = float(module.momentum)
            module.running_mean = (1.0 - momentum) * module.running_mean + momentum * mean.reshape(-1)
            module.running_var = (1.0 - momentum) * module.running_var + momentum * var.reshape(-1)
    else:
        mean = module.running_mean.reshape(stat_shape)
        var = module.running_var.reshape(stat_shape)

    invstd = 1.0 / np.sqrt(var + float(module.eps))
    xhat = (x_np - mean) * invstd

    if module.weight is not None:
        w_b = module.weight.numpy().reshape(stat_shape)
        y_np = xhat * w_b
    else:
        w_b = None
        y_np = xhat
    if module.bias is not None:
        b_b = module.bias.numpy().reshape(stat_shape)
        y_np = y_np + b_b

    track = is_grad_enabled() and (
        x.requires_grad
        or (module.weight is not None and module.weight.requires_grad)
        or (module.bias is not None and module.bias.requires_grad)
    )
    out = _make_tensor_from_np(y_np, device=x.device, requires_grad=track, op=op_name)
    if not track:
        return out

    out._prev = {x}
    norm_size = float(np.prod([x_np.shape[axis] for axis in reduce_axes]))

    def _backward() -> None:
        grad_out = _grad_from_output(out)
        if grad_out is None:
            return

        dxhat = grad_out
        if module.weight is not None:
            if module.weight.requires_grad:
                dgamma = (grad_out * xhat).sum(axis=reduce_axes, keepdims=False)
                _accum_tensor_grad(module.weight, dgamma)
            dxhat = grad_out * w_b
        if module.bias is not None and module.bias.requires_grad:
            dbeta = grad_out.sum(axis=reduce_axes, keepdims=False)
            _accum_tensor_grad(module.bias, dbeta)

        if x.requires_grad:
            sum1 = dxhat.sum(axis=reduce_axes, keepdims=True)
            sum2 = (dxhat * xhat).sum(axis=reduce_axes, keepdims=True)
            dx = (invstd / max(norm_size, 1.0)) * (norm_size * dxhat - sum1 - xhat * sum2)
            _accum_tensor_grad(x, dx)

    out._backward = _backward
    return out


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
        self.weight = Parameter(np.empty((out_features, in_features), dtype=np.float32))
        init_ops.kaiming_normal_(self.weight, nonlinearity="relu")
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

        self.weight = Parameter(np.empty((self.out_channels, self.in_channels, kh, kw), dtype=np.float32))
        init_ops.kaiming_normal_(self.weight, nonlinearity="relu")
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


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x.gelu()


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x.silu()


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x.leaky_relu(self.negative_slope)


class ELU(Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x.elu(self.alpha)


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


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(self.kernel_size if stride is None else stride)
        self.padding = _pair(padding)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if len(x.shape) != 4:
            raise ValueError(f"MaxPool2d expects shape [N,C,H,W], got {x.shape}")

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        if kh <= 0 or kw <= 0 or sh <= 0 or sw <= 0:
            raise ValueError("kernel_size and stride must be > 0")

        x_np = np.asarray(x.numpy(), dtype=np.float32)
        x_pad = np.pad(x_np, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant", constant_values=-np.inf)
        n, c, hp, wp = x_pad.shape
        oh = (hp - kh) // sh + 1
        ow = (wp - kw) // sw + 1

        out_np = np.empty((n, c, oh, ow), dtype=np.float32)
        max_idx = np.empty((n, c, oh, ow), dtype=np.int64)
        for i in range(oh):
            hs = i * sh
            for j in range(ow):
                ws = j * sw
                window = x_pad[:, :, hs : hs + kh, ws : ws + kw].reshape(n, c, kh * kw)
                idx = np.argmax(window, axis=2)
                out_np[:, :, i, j] = np.take_along_axis(window, idx[:, :, None], axis=2)[:, :, 0]
                max_idx[:, :, i, j] = idx

        track = is_grad_enabled() and x.requires_grad
        out = _make_tensor_from_np(out_np, device=x.device, requires_grad=track, op="max_pool2d")
        if not track:
            return out

        out._prev = {x}

        def _backward() -> None:
            grad_out = _grad_from_output(out)
            if grad_out is None:
                return

            dx_pad = np.zeros_like(x_pad, dtype=np.float32)
            n_idx = np.arange(n)[:, None]
            c_idx = np.arange(c)[None, :]
            for i in range(oh):
                hs = i * sh
                for j in range(ow):
                    ws = j * sw
                    idx = max_idx[:, :, i, j]
                    np.add.at(
                        dx_pad,
                        (n_idx, c_idx, hs + (idx // kw), ws + (idx % kw)),
                        grad_out[:, :, i, j],
                    )

            dx = dx_pad[:, :, ph : ph + x.shape[2], pw : pw + x.shape[3]]
            _accum_tensor_grad(x, dx)

        out._backward = _backward
        return out


class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(self.kernel_size if stride is None else stride)
        self.padding = _pair(padding)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if len(x.shape) != 4:
            raise ValueError(f"AvgPool2d expects shape [N,C,H,W], got {x.shape}")

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        if kh <= 0 or kw <= 0 or sh <= 0 or sw <= 0:
            raise ValueError("kernel_size and stride must be > 0")

        x_np = np.asarray(x.numpy(), dtype=np.float32)
        x_pad = np.pad(x_np, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant", constant_values=0.0)
        n, c, hp, wp = x_pad.shape
        oh = (hp - kh) // sh + 1
        ow = (wp - kw) // sw + 1

        out_np = np.empty((n, c, oh, ow), dtype=np.float32)
        for i in range(oh):
            hs = i * sh
            for j in range(ow):
                ws = j * sw
                window = x_pad[:, :, hs : hs + kh, ws : ws + kw]
                out_np[:, :, i, j] = window.mean(axis=(2, 3))

        track = is_grad_enabled() and x.requires_grad
        out = _make_tensor_from_np(out_np, device=x.device, requires_grad=track, op="avg_pool2d")
        if not track:
            return out

        out._prev = {x}
        scale = 1.0 / float(kh * kw)

        def _backward() -> None:
            grad_out = _grad_from_output(out)
            if grad_out is None:
                return

            dx_pad = np.zeros_like(x_pad, dtype=np.float32)
            for i in range(oh):
                hs = i * sh
                for j in range(ow):
                    ws = j * sw
                    dx_pad[:, :, hs : hs + kh, ws : ws + kw] += grad_out[:, :, i : i + 1, j : j + 1] * scale

            dx = dx_pad[:, :, ph : ph + x.shape[2], pw : pw + x.shape[3]]
            _accum_tensor_grad(x, dx)

        out._backward = _backward
        return out


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
        if x.device == "gpu" and abs(self.eps - 1e-5) <= 1e-12:
            if len(self.normalized_shape) != 1:
                raise NotImplementedError("LayerNorm GPU path currently supports 1D normalized_shape only")
            if len(x.shape) != 2:
                raise NotImplementedError("LayerNorm GPU path currently supports 2D inputs only")
            if tuple(x.shape[-1:]) != tuple(self.normalized_shape):
                raise ValueError(
                    f"LayerNorm normalized_shape mismatch: expected trailing {self.normalized_shape}, got {tuple(x.shape)}"
                )

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

        out = _make_tensor_from_np(y.astype(np.float32, copy=False), device=x.device, requires_grad=track, op="layer_norm")
        if not track:
            return out

        out._prev = {x}

        # Precompute for backward
        N = float(np.prod(self.normalized_shape))
        param_axes = tuple(range(0, xhat.ndim - len(self.normalized_shape)))

        def _backward() -> None:
            grad_out = _grad_from_output(out)
            if grad_out is None:
                return
            dy = grad_out

            # grads for affine
            dxhat = dy
            if self.weight is not None:
                w_b = w  # broadcasted
                if self.weight.requires_grad:
                    dgamma = (dy * xhat).sum(axis=param_axes, keepdims=False)
                    _accum_tensor_grad(self.weight, dgamma)
                dxhat = dy * w_b
            if self.bias is not None and self.bias.requires_grad:
                dbeta = dy.sum(axis=param_axes, keepdims=False)
                _accum_tensor_grad(self.bias, dbeta)

            if x.requires_grad:
                # dx = (1/N)*invstd*(N*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
                sum1 = dxhat.sum(axis=axes, keepdims=True)
                sum2 = (dxhat * xhat).sum(axis=axes, keepdims=True)
                dx = (invstd / max(N, 1.0)) * (dxhat * N - sum1 - xhat * sum2)
                _accum_tensor_grad(x, dx)

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


class BatchNorm1d(Module):
    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.track_running_stats = bool(track_running_stats)
        self.running_mean = np.zeros((self.num_features,), dtype=np.float32)
        self.running_var = np.ones((self.num_features,), dtype=np.float32)
        self.weight = Parameter(np.ones((self.num_features,), dtype=np.float32)) if affine else None
        self.bias = Parameter(np.zeros((self.num_features,), dtype=np.float32)) if affine else None

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if len(x.shape) == 2:
            reduce_axes = (0,)
        elif len(x.shape) == 3:
            reduce_axes = (0, 2)
        else:
            raise ValueError(f"BatchNorm1d expects shape [N,C] or [N,C,L], got {x.shape}")
        if x.shape[1] != self.num_features:
            raise ValueError(f"BatchNorm1d expected channel dimension {self.num_features}, got {x.shape[1]}")
        return _batchnorm_forward(self, x, reduce_axes, "batch_norm1d")


class BatchNorm2d(Module):
    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.track_running_stats = bool(track_running_stats)
        self.running_mean = np.zeros((self.num_features,), dtype=np.float32)
        self.running_var = np.ones((self.num_features,), dtype=np.float32)
        self.weight = Parameter(np.ones((self.num_features,), dtype=np.float32)) if affine else None
        self.bias = Parameter(np.zeros((self.num_features,), dtype=np.float32)) if affine else None

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if len(x.shape) != 4:
            raise ValueError(f"BatchNorm2d expects shape [N,C,H,W], got {x.shape}")
        if x.shape[1] != self.num_features:
            raise ValueError(f"BatchNorm2d expected channel dimension {self.num_features}, got {x.shape[1]}")
        return _batchnorm_forward(self, x, (0, 2, 3), "batch_norm2d")


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight = Parameter(np.empty((self.num_embeddings, self.embedding_dim), dtype=np.float32)).to(device)
        init_ops.normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, indices: Tensor | np.ndarray | Sequence[int]) -> Tensor:  # type: ignore[override]
        if isinstance(indices, Tensor):
            idx = np.asarray(indices.numpy(), dtype=np.int64)
            out_device = self.weight.device if self.weight.device == "gpu" or indices.device == "gpu" else "cpu"
        else:
            idx = np.asarray(indices, dtype=np.int64)
            out_device = self.weight.device

        if idx.size > 0 and (idx.min() < 0 or idx.max() >= self.num_embeddings):
            raise ValueError(f"Embedding indices out of range [0,{self.num_embeddings})")

        y_np = self.weight.numpy()[idx]
        track = is_grad_enabled() and self.weight.requires_grad
        out = _make_tensor_from_np(y_np, device=out_device, requires_grad=track, op="embedding")

        if track:
            flat_idx = idx.reshape(-1)

            def _backward() -> None:
                grad_out = _grad_from_output(out)
                if grad_out is None:
                    return
                grad_weight = np.zeros_like(self.weight.numpy(), dtype=np.float32)
                np.add.at(grad_weight, flat_idx, grad_out.reshape(-1, self.embedding_dim))
                _accum_tensor_grad(self.weight, grad_weight)

            out._backward = _backward

        return out


class MultiheadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, batch_first: bool = True) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be > 0, num_heads must be > 0, and embed_dim must be divisible by num_heads")
        if not batch_first:
            raise NotImplementedError("MultiheadAttention currently supports batch_first=True only")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = True

        self.q_weight = Parameter(np.empty((self.embed_dim, self.embed_dim), dtype=np.float32))
        self.k_weight = Parameter(np.empty((self.embed_dim, self.embed_dim), dtype=np.float32))
        self.v_weight = Parameter(np.empty((self.embed_dim, self.embed_dim), dtype=np.float32))
        self.out_weight = Parameter(np.empty((self.embed_dim, self.embed_dim), dtype=np.float32))
        self.q_bias = Parameter(np.zeros((self.embed_dim,), dtype=np.float32))
        self.k_bias = Parameter(np.zeros((self.embed_dim,), dtype=np.float32))
        self.v_bias = Parameter(np.zeros((self.embed_dim,), dtype=np.float32))
        self.out_bias = Parameter(np.zeros((self.embed_dim,), dtype=np.float32))

        init_ops.xavier_uniform_(self.q_weight)
        init_ops.xavier_uniform_(self.k_weight)
        init_ops.xavier_uniform_(self.v_weight)
        init_ops.xavier_uniform_(self.out_weight)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
    ):
        if len(query.shape) != 3 or len(key.shape) != 3 or len(value.shape) != 3:
            raise ValueError("MultiheadAttention expects query, key, value with shape [B,T,E]")
        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            raise ValueError("MultiheadAttention expects matching batch sizes")
        if key.shape[1] != value.shape[1]:
            raise ValueError("MultiheadAttention expects key and value sequence lengths to match")
        if query.shape[2] != self.embed_dim or key.shape[2] != self.embed_dim or value.shape[2] != self.embed_dim:
            raise ValueError(f"MultiheadAttention expects embed_dim={self.embed_dim}")

        q_np = np.asarray(query.numpy(), dtype=np.float32)
        k_np = np.asarray(key.numpy(), dtype=np.float32)
        v_np = np.asarray(value.numpy(), dtype=np.float32)
        bsz, tgt_len, _ = q_np.shape
        src_len = k_np.shape[1]
        scale = float(np.sqrt(self.head_dim))

        q_lin = q_np @ self.q_weight.numpy().T + self.q_bias.numpy()
        k_lin = k_np @ self.k_weight.numpy().T + self.k_bias.numpy()
        v_lin = v_np @ self.v_weight.numpy().T + self.v_bias.numpy()

        q_heads = q_lin.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k_heads = k_lin.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v_heads = v_lin.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(q_heads, np.swapaxes(k_heads, -1, -2)) / scale
        scores_max = scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores - scores_max)
        attn = attn / np.maximum(attn.sum(axis=-1, keepdims=True), 1e-20)
        context = np.matmul(attn, v_heads)
        context_merged = context.transpose(0, 2, 1, 3).reshape(bsz, tgt_len, self.embed_dim)
        out_np = context_merged @ self.out_weight.numpy().T + self.out_bias.numpy()

        device = "gpu" if any(t.device == "gpu" for t in (query, key, value, self.q_weight)) else "cpu"
        track = is_grad_enabled() and any(
            t.requires_grad for t in (query, key, value, self.q_weight, self.k_weight, self.v_weight, self.out_weight, self.q_bias, self.k_bias, self.v_bias, self.out_bias)
        )
        out = _make_tensor_from_np(out_np, device=device, requires_grad=track, op="multihead_attention")

        if track:
            out._prev = {query, key, value}

            def _backward() -> None:
                grad_out = _grad_from_output(out)
                if grad_out is None:
                    return

                grad_out_2d = grad_out.reshape(bsz * tgt_len, self.embed_dim)
                context_2d = context_merged.reshape(bsz * tgt_len, self.embed_dim)
                _accum_tensor_grad(self.out_weight, grad_out_2d.T @ context_2d)
                _accum_tensor_grad(self.out_bias, grad_out_2d.sum(axis=0))

                d_context_merged = grad_out_2d @ self.out_weight.numpy()
                d_context = d_context_merged.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

                d_attn = np.matmul(d_context, np.swapaxes(v_heads, -1, -2))
                d_v_heads = np.matmul(np.swapaxes(attn, -1, -2), d_context)

                d_scores = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))
                d_scores = d_scores / scale
                d_q_heads = np.matmul(d_scores, k_heads)
                d_k_heads = np.matmul(np.swapaxes(d_scores, -1, -2), q_heads)

                d_q_lin = d_q_heads.transpose(0, 2, 1, 3).reshape(bsz, tgt_len, self.embed_dim)
                d_k_lin = d_k_heads.transpose(0, 2, 1, 3).reshape(bsz, src_len, self.embed_dim)
                d_v_lin = d_v_heads.transpose(0, 2, 1, 3).reshape(bsz, src_len, self.embed_dim)

                q_2d = q_np.reshape(bsz * tgt_len, self.embed_dim)
                k_2d = k_np.reshape(bsz * src_len, self.embed_dim)
                v_2d = v_np.reshape(bsz * src_len, self.embed_dim)
                dq_2d = d_q_lin.reshape(bsz * tgt_len, self.embed_dim)
                dk_2d = d_k_lin.reshape(bsz * src_len, self.embed_dim)
                dv_2d = d_v_lin.reshape(bsz * src_len, self.embed_dim)

                _accum_tensor_grad(self.q_weight, dq_2d.T @ q_2d)
                _accum_tensor_grad(self.k_weight, dk_2d.T @ k_2d)
                _accum_tensor_grad(self.v_weight, dv_2d.T @ v_2d)
                _accum_tensor_grad(self.q_bias, dq_2d.sum(axis=0))
                _accum_tensor_grad(self.k_bias, dk_2d.sum(axis=0))
                _accum_tensor_grad(self.v_bias, dv_2d.sum(axis=0))

                if query.requires_grad:
                    _accum_tensor_grad(query, (dq_2d @ self.q_weight.numpy()).reshape(bsz, tgt_len, self.embed_dim))
                if key.requires_grad:
                    _accum_tensor_grad(key, (dk_2d @ self.k_weight.numpy()).reshape(bsz, src_len, self.embed_dim))
                if value.requires_grad:
                    _accum_tensor_grad(value, (dv_2d @ self.v_weight.numpy()).reshape(bsz, src_len, self.embed_dim))

            out._backward = _backward

        if not need_weights:
            return out
        weights = _make_tensor_from_np(attn.mean(axis=1), device=device, requires_grad=False, op="attention_weights")
        return out, weights


class GRU(Module):
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be > 0")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_first = bool(batch_first)

        self.weight_ih_l0 = Parameter(np.empty((3 * self.hidden_size, self.input_size), dtype=np.float32))
        self.weight_hh_l0 = Parameter(np.empty((3 * self.hidden_size, self.hidden_size), dtype=np.float32))
        self.bias_ih_l0 = Parameter(np.zeros((3 * self.hidden_size,), dtype=np.float32))
        self.bias_hh_l0 = Parameter(np.zeros((3 * self.hidden_size,), dtype=np.float32))
        init_ops.xavier_uniform_(self.weight_ih_l0)
        init_ops.xavier_uniform_(self.weight_hh_l0)

    def forward(self, x: Tensor, h0: Tensor | None = None):
        if len(x.shape) != 3:
            raise ValueError(f"GRU expects a 3D input, got {x.shape}")

        x_np = np.asarray(x.numpy(), dtype=np.float32)
        if self.batch_first:
            batch_size, seq_len, input_size = x_np.shape
        else:
            seq_len, batch_size, input_size = x_np.shape
            x_np = np.transpose(x_np, (1, 0, 2))
        if input_size != self.input_size:
            raise ValueError(f"GRU expected input_size {self.input_size}, got {input_size}")

        if h0 is None:
            h = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        else:
            h0_np = np.asarray(h0.numpy(), dtype=np.float32)
            if h0_np.shape != (1, batch_size, self.hidden_size):
                raise ValueError(f"GRU expected h0 shape (1, {batch_size}, {self.hidden_size}), got {h0_np.shape}")
            h = h0_np[0]

        w_ih = self.weight_ih_l0.numpy()
        w_hh = self.weight_hh_l0.numpy()
        b_ih = self.bias_ih_l0.numpy()
        b_hh = self.bias_hh_l0.numpy()

        outputs = []
        cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for t in range(seq_len):
            xt = x_np[:, t, :]
            h_prev = h.copy()
            gi = xt @ w_ih.T + b_ih
            gh = h @ w_hh.T + b_hh
            i_r, i_z, i_n = np.split(gi, 3, axis=1)
            h_r, h_z, h_n = np.split(gh, 3, axis=1)
            r = 1.0 / (1.0 + np.exp(-(i_r + h_r)))
            z = 1.0 / (1.0 + np.exp(-(i_z + h_z)))
            n = np.tanh(i_n + r * h_n)
            h = (1.0 - z) * n + z * h
            outputs.append(h.copy())
            cache.append((xt, h_prev, r, z, n, h_n))

        output_np = np.stack(outputs, axis=1)
        if not self.batch_first:
            output_np = np.transpose(output_np, (1, 0, 2))
        h_n = h.reshape(1, batch_size, self.hidden_size)

        device = x.device if x.device == "gpu" or self.weight_ih_l0.device == "gpu" else "cpu"
        track = is_grad_enabled() and (
            x.requires_grad
            or (h0 is not None and h0.requires_grad)
            or self.weight_ih_l0.requires_grad
            or self.weight_hh_l0.requires_grad
            or self.bias_ih_l0.requires_grad
            or self.bias_hh_l0.requires_grad
        )
        output = _make_tensor_from_np(output_np, device=device, requires_grad=track, op="gru")
        hidden = _make_tensor_from_np(h_n, device=device, requires_grad=track, op="gru_hidden")

        if track:
            parents = {x}
            if h0 is not None:
                parents.add(h0)
            output._prev = parents
            hidden._prev = parents

            def _accumulate_grads(grad_output: np.ndarray | None, grad_hidden: np.ndarray | None) -> None:
                local_grad_output = (
                    np.zeros((batch_size, seq_len, self.hidden_size), dtype=np.float32)
                    if grad_output is None
                    else np.asarray(grad_output, dtype=np.float32)
                )
                if not self.batch_first:
                    local_grad_output = np.transpose(local_grad_output, (1, 0, 2))
                dh_next = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
                if grad_hidden is not None:
                    dh_next = dh_next + np.asarray(grad_hidden, dtype=np.float32)[0]

                dx = np.zeros_like(x_np, dtype=np.float32)
                dw_ih = np.zeros_like(w_ih, dtype=np.float32)
                dw_hh = np.zeros_like(w_hh, dtype=np.float32)
                db_ih = np.zeros_like(b_ih, dtype=np.float32)
                db_hh = np.zeros_like(b_hh, dtype=np.float32)

                for t in range(seq_len - 1, -1, -1):
                    xt, h_prev, r, z, n, h_n_part = cache[t]
                    dh = dh_next + local_grad_output[:, t, :]
                    dz = (h_prev - n) * dh
                    dn = (1.0 - z) * dh
                    dh_carry = z * dh

                    d_inner_n = (1.0 - n ** 2) * dn
                    di_n = d_inner_n
                    dh_n = r * d_inner_n
                    dr = h_n_part * d_inner_n

                    d_inner_z = z * (1.0 - z) * dz
                    di_z = d_inner_z
                    dh_z = d_inner_z

                    d_inner_r = r * (1.0 - r) * dr
                    di_r = d_inner_r
                    dh_r = d_inner_r

                    dgi = np.concatenate([di_r, di_z, di_n], axis=1)
                    dgh = np.concatenate([dh_r, dh_z, dh_n], axis=1)

                    dx[:, t, :] = dgi @ w_ih
                    dw_ih += dgi.T @ xt
                    dw_hh += dgh.T @ h_prev
                    db_ih += dgi.sum(axis=0)
                    db_hh += dgh.sum(axis=0)
                    dh_next = dh_carry + dgh @ w_hh

                dx_out = dx if self.batch_first else np.transpose(dx, (1, 0, 2))
                _accum_tensor_grad(x, dx_out)
                if h0 is not None:
                    _accum_tensor_grad(h0, dh_next.reshape(1, batch_size, self.hidden_size))
                _accum_tensor_grad(self.weight_ih_l0, dw_ih)
                _accum_tensor_grad(self.weight_hh_l0, dw_hh)
                _accum_tensor_grad(self.bias_ih_l0, db_ih)
                _accum_tensor_grad(self.bias_hh_l0, db_hh)

            def _backward_output() -> None:
                grad_output = _grad_from_output(output)
                if grad_output is None:
                    return
                _accumulate_grads(grad_output, None)

            def _backward_hidden() -> None:
                grad_hidden = _grad_from_output(hidden)
                if grad_hidden is None:
                    return
                _accumulate_grads(None, grad_hidden)

            output._backward = _backward_output
            hidden._backward = _backward_hidden
        return output, hidden
