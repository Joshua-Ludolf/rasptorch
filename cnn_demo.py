import numpy as np
from rasptorch import functional as F
from rasptorch import vulkan_backend as vk
from rasptorch.data import DataLoader, TensorDataset
from rasptorch.train import Accuracy, classification_target_one_hot, fit
from rasptorch.nn import Conv2d, Flatten, Linear, ReLU, Sequential
from rasptorch.optim import SGD
from rasptorch.tensor import Tensor


def make_spot_dataset(n: int, *, num_classes: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A tiny, learnable classification dataset.

    Each class corresponds to a fixed (row,col) "spot" in the 28x28 image.
    The image is mostly noise with one bright pixel at the class spot.
    """

    rng = np.random.default_rng(seed)
    # 10 distinct spots
    spots = [
        (4, 4),
        (4, 14),
        (4, 24),
        (10, 6),
        (10, 22),
        (14, 14),
        (18, 6),
        (18, 22),
        (24, 4),
        (24, 24),
    ]
    if num_classes != 10:
        raise ValueError("make_spot_dataset currently supports num_classes=10")

    y = rng.integers(0, num_classes, size=(n,), dtype=np.int64)
    x = (0.05 * rng.standard_normal((n, 1, 28, 28))).astype(np.float32)
    for i, cls in enumerate(y):
        r, c = spots[int(cls)]
        x[i, 0, r, c] += 1.0
    return x, y


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    pred = logits.argmax(axis=1)
    return float((pred == labels).mean())


def main() -> None:
    vk.init(strict=True)

    # Tiny CNN -> classifier head.
    model = Sequential(
        Conv2d(1, 4, 3, stride=1, padding=0),
        ReLU(),
        Flatten(),
        Linear(4 * 26 * 26, 10),
    ).to("gpu")

    # Conservative LR; momentum helps training stability.
    opt = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # Data
    x_train, y_train = make_spot_dataset(1024, seed=0)
    x_val, y_val = make_spot_dataset(256, seed=1)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)

    fit(
        model,
        opt,
        train_loader,
        loss_fn=F.cross_entropy,
        device="gpu",
        epochs=10,
        val_loader=val_loader,
        target_transform=classification_target_one_hot(10, device="gpu"),
        metrics=[Accuracy()],
    )


if __name__ == "__main__":
    main()
