import argparse
import numpy as np

from rasptorch.data import DataLoader, TensorDataset
from rasptorch.gpu_training import train_mlp_regression_gpu
from rasptorch.nn import Linear, ReLU, Sequential
from rasptorch.optim import SGD
from rasptorch.tensor import Tensor
from rasptorch.functional import mse_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="rasptorch demo")
    parser.add_argument("--device", choices=["cpu", "gpu", "gpu-autograd"], default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save", type=str, default="", help="Save trained weights to a .pth file")
    args = parser.parse_args()

    # Generate synthetic data: y = 2x^2 + 1 (nonlinear)
    x = np.linspace(-1, 1, 200, dtype="float32").reshape(-1, 1)
    y = 2 * (x ** 2) + 1

    if args.device == "gpu":
        try:
            train_mlp_regression_gpu(
                x,
                y,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                log_every=args.log_every,
                save_path=(args.save or None),
            )
        except RuntimeError as e:
            print("Vulkan GPU mode failed:")
            print(f"  {e}")
            print(
                "Tip: ensure Vulkan is installed and working, the Python 'vulkan' package is available, "
                "and 'glslc' (shader compiler) is installed on the system."
            )
            raise SystemExit(1) from e

        print("Training done.")
        return

    x_t = Tensor(x, requires_grad=False)
    y_t = Tensor(y, requires_grad=False)

    dataset = TensorDataset(x_t.data, y_t.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # A slightly deeper model using Sequential + ReLU
    model = Sequential(
        Linear(1, 16),
        ReLU(),
        Linear(16, 1),
    )
    if args.device == "gpu-autograd":
        model.to("gpu")
    optimizer = SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for xb_np, yb_np in loader:
            xb = Tensor(xb_np, requires_grad=False)
            yb = Tensor(yb_np, requires_grad=False)
            if args.device == "gpu-autograd":
                xb = xb.to("gpu")
                yb = yb.to("gpu")

            optimizer.zero_grad()
            preds = model(xb)
            loss = mse_loss(preds, yb)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.numpy().reshape(-1)[0])
            epoch_loss += loss_val
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={avg_loss:.6f}")

    model.eval()

    if args.save:
        payload = {"arch": type(model).__name__, "state_dict": model.state_dict()}
        try:
            import torch  # type: ignore

            torch.save(payload, args.save)
        except ModuleNotFoundError:
            # Fallback: still write a file, but it won't be torch.load()-compatible.
            import pickle

            with open(args.save, "wb") as f:
                pickle.dump(payload, f)
            print("Warning: 'torch' not installed; wrote a pickle file (not torch.load compatible).")
        print(f"Saved model to: {args.save}")

    print("Training done.")


if __name__ == "__main__":
    main()
