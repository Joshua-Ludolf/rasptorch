import numpy as np

from rasptorch import Tensor, no_grad
from rasptorch import functional as F


def test_no_grad_blocks_graph_wiring() -> None:
    x = Tensor(np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    w = Tensor(np.array([[2.0, 1.0], [-1.0, 0.5]], dtype=np.float32), requires_grad=True)

    with no_grad():
        y = x * 2.0
        assert y.requires_grad is False

    # y should behave as a constant; gradient should flow only to w.
    z = (y @ w).sum()
    assert z.requires_grad is True
    z.backward()

    assert x.grad is None
    assert w.grad is not None


def test_detach_blocks_grad_flow() -> None:
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    y = x.detach()
    assert y.requires_grad is False

    z = (y * 3.0).sum()
    assert z.requires_grad is False
    z.backward()
    assert x.grad is None


def test_no_grad_cross_entropy_requires_grad_false() -> None:
    rng = np.random.default_rng(0)
    logits = Tensor(rng.standard_normal((4, 5), dtype=np.float32), requires_grad=True)
    target = F.one_hot([0, 1, 2, 3], 5)

    with no_grad():
        loss = F.cross_entropy(logits, target)
        assert loss.requires_grad is False
