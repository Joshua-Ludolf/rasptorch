from importlib.metadata import PackageNotFoundError, version

from .tensor import Tensor, Parameter, no_grad, enable_grad, set_grad_enabled, is_grad_enabled
from . import nn, functional, data, train
from . import torch_bridge

try:
	__version__ = version("rasptorch")
except PackageNotFoundError:  # pragma: no cover
	__version__ = "1.1.0"

__all__ = [
	"Tensor",
	"Parameter",
	"no_grad",
	"enable_grad",
	"set_grad_enabled",
	"is_grad_enabled",
	"__version__",
	"nn",
	"functional",
	"data",
	"train",
	"torch_bridge",
]
