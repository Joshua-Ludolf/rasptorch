from importlib.metadata import PackageNotFoundError, version

from .tensor import Tensor, Parameter
from . import nn, functional, data, train

try:
	__version__ = version("rasptorch")
except PackageNotFoundError:  # pragma: no cover
	__version__ = "0.0.0"

__all__ = [
	"Tensor",
	"Parameter",
	"__version__",
	"nn",
	"functional",
	"data",
	"train",
]
