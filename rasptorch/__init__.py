from importlib.metadata import PackageNotFoundError, version

from .tensor import Tensor, Parameter, cat, stack, no_grad, enable_grad, set_grad_enabled, is_grad_enabled
from .optim import SGD, Adam, AdamW, RMSProp
from .optim_sched import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, WarmupScheduler
from .amp import autocast, GradScaler
from .gpu_training import GpuMLP
from . import nn, functional, data, train, optim, optim_sched, init, utils, amp, data_validation
from . import torch_bridge


try:
	__version__ = version("rasptorch")
except PackageNotFoundError:  # pragma: no cover
	__version__ = "latest"

__all__ = [
	"Tensor",
	"Parameter",
	"cat",
	"stack",
	"no_grad",
	"enable_grad",
	"set_grad_enabled",
	"is_grad_enabled",
	"SGD",
	"Adam",
	"AdamW",
	"RMSProp",
	"StepLR",
	"MultiStepLR",
	"ExponentialLR",
	"CosineAnnealingLR",
	"ReduceLROnPlateau",
	"WarmupScheduler",
	"autocast",
	"GradScaler",
	"GpuMLP",
	"__version__",
	"nn",
	"functional",
	"data",
	"train",
	"optim",
	"optim_sched",
	"init",
	"utils",
	"amp",
	"data_validation",
	"torch_bridge",
]
