import numpy as np

from rasptorch.data import DataLoader, TensorDataset
from rasptorch.nn import Linear, ReLU, Sequential
from rasptorch.optim import SGD
from rasptorch.tensor import Tensor
from rasptorch.functional import mse_loss


def main() -> None:
    # Generate synthetic data: y = 2x^2 + 1 (nonlinear)
    x = np.linspace(-1, 1, 200, dtype="float32").reshape(-1, 1)
    y = 2 * (x ** 2) + 1

    x_t = Tensor(x, requires_grad=False)
    y_t = Tensor(y, requires_grad=False)

    dataset = TensorDataset(x_t.data, y_t.data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # A slightly deeper model using Sequential + ReLU
    model = Sequential(
        Linear(1, 16),
        ReLU(),
        Linear(16, 1),
    )
    optimizer = SGD(model.parameters(), lr=0.1)

    for epoch in range(50):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for xb_np, yb_np in loader:
            xb = Tensor(xb_np, requires_grad=False)
            yb = Tensor(yb_np, requires_grad=False)

            optimizer.zero_grad()
            preds = model(xb)
            loss = mse_loss(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data)
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={avg_loss:.6f}")

    model.eval()
    print("Training done.")


if __name__ == "__main__":
    main()
