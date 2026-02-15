import numpy as np

from rasptorch.data import DataLoader, TensorDataset
from rasptorch import functional as F
from rasptorch.nn import LayerNorm, Linear, ReLU, Sequential
from rasptorch.optim import SGD
from rasptorch.train import Accuracy, classification_target_one_hot, evaluate, train_one_epoch


def test_train_one_epoch_runs_on_gpu_device() -> None:
    rng = np.random.default_rng(0)
    N, Din, C = 64, 10, 4

    x = rng.standard_normal((N, Din), dtype=np.float32)
    y = rng.integers(0, C, size=(N,), dtype=np.int64)

    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    model = Sequential(
        Linear(Din, 16),
        LayerNorm(16),
        ReLU(),
        Linear(16, C),
    ).to("gpu")

    opt = SGD(model.parameters(), lr=0.05)
    tgt = classification_target_one_hot(C, device="gpu")

    stats = train_one_epoch(
        model,
        opt,
        loader,
        loss_fn=F.cross_entropy,
        device="gpu",
        target_transform=tgt,
        metrics=[Accuracy()],
    )

    assert np.isfinite(stats.loss)
    assert 0.0 <= stats.metrics["acc"] <= 1.0

    # Evaluate path should run under no_grad (smoke check)
    ev = evaluate(
        model,
        loader,
        loss_fn=F.cross_entropy,
        device="gpu",
        target_transform=tgt,
        metrics=[Accuracy()],
    )
    assert np.isfinite(ev.loss)
    assert 0.0 <= ev.metrics["acc"] <= 1.0
