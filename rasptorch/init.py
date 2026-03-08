from __future__ import annotations

import math

import numpy as np

from . import vulkan_backend as vk
from .tensor import Tensor


def _shape_of(tensor: Tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape)


def _assign(tensor: Tensor, values: np.ndarray) -> Tensor:
    arr = np.asarray(values, dtype=np.float32).reshape(_shape_of(tensor))
    if tensor.device == "gpu":
        if tensor._vkbuf is None:
            raise RuntimeError("GPU tensor is missing its Vulkan buffer")
        vk.write(tensor._vkbuf, arr)
        tensor.data = np.empty(arr.shape, dtype=np.float32)
    else:
        tensor.data = arr
    return tensor


def _calculate_fan_in_and_fan_out(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return shape[0], shape[0]

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if len(shape) > 2:
        receptive_field_size = int(np.prod(shape[2:]))
    fan_in = int(num_input_fmaps * receptive_field_size)
    fan_out = int(num_output_fmaps * receptive_field_size)
    return fan_in, fan_out


def _calculate_gain(nonlinearity: str, a: float = 0.0) -> float:
    table = {
        "linear": 1.0,
        "conv1d": 1.0,
        "conv2d": 1.0,
        "conv3d": 1.0,
        "sigmoid": 1.0,
        "tanh": 5.0 / 3.0,
        "relu": math.sqrt(2.0),
    }
    if nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1.0 + a ** 2))
    return table.get(nonlinearity, 1.0)


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    values = np.random.uniform(float(a), float(b), size=_shape_of(tensor)).astype(np.float32)
    return _assign(tensor, values)


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    values = np.random.normal(float(mean), float(std), size=_shape_of(tensor)).astype(np.float32)
    return _assign(tensor, values)


def constant_(tensor: Tensor, val: float) -> Tensor:
    values = np.full(_shape_of(tensor), float(val), dtype=np.float32)
    return _assign(tensor, values)


def zeros_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 0.0)


def ones_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 1.0)


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(_shape_of(tensor))
    bound = float(gain) * math.sqrt(6.0 / max(1, fan_in + fan_out))
    return uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(_shape_of(tensor))
    std = float(gain) * math.sqrt(2.0 / max(1, fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(_shape_of(tensor))
    fan = fan_in if mode == "fan_in" else fan_out
    gain = _calculate_gain(nonlinearity, a)
    bound = math.sqrt(3.0) * gain / math.sqrt(max(1, fan))
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(_shape_of(tensor))
    fan = fan_in if mode == "fan_in" else fan_out
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(max(1, fan))
    return normal_(tensor, 0.0, std)


def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    shape = _shape_of(tensor)
    if len(shape) < 2:
        raise ValueError("orthogonal_ requires a tensor with at least 2 dimensions")

    rows = shape[0]
    cols = int(np.prod(shape[1:]))
    flat = np.random.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32)
    if rows < cols:
        flat = flat.T
    q, r = np.linalg.qr(flat)
    d = np.diag(r)
    q *= np.sign(d)
    if rows < cols:
        q = q.T
    q = q.reshape(shape)
    return _assign(tensor, float(gain) * q.astype(np.float32))