import argparse
import numpy as np

from rasptorch.data import DataLoader, TensorDataset
from rasptorch.gpu_training import BackendMLP, train_mlp_regression_gpu
from rasptorch.nn import Linear, ReLU, Sequential
from rasptorch.optim import SGD
from rasptorch.tensor import Tensor
from rasptorch.functional import mse_loss
from rasptorch.checkpoint import save_checkpoint
from rasptorch import connect_backend


def main() -> None:
    '''A simple demo showing how to train a small MLP on synthetic regression data, using either CPU or Vulkan GPU backends.'''
    parser = argparse.ArgumentParser(description="rasptorch demo")
    parser.add_argument("--device", choices=["cpu", "gpu", "gpu-autograd", "auto"], default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible CPU/GPU comparisons")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save", type=str, default="", help="Save trained weights to a .pth file")
    args = parser.parse_args()
    np.random.seed(args.seed)

    # Generate synthetic data: y = 2x^2 + 1 (nonlinear)
    x = np.linspace(-1, 1, 200, dtype="float32").reshape(-1, 1)
    y = 2 * (x ** 2) + 1

    if args.device in {"gpu", "auto"}:
        # Prefer Vulkan, but fall back to OpenCL/CUDA if Vulkan isn't available.
        selected_backend = connect_backend("vulkan", strict=False)
        if selected_backend.name != "vulkan":
            selected_backend = connect_backend("opencl", strict=False)
        if selected_backend.name not in {"vulkan", "opencl"}:
            selected_backend = connect_backend("cuda", strict=False)
    else:
        selected_backend = connect_backend("cpu", strict=False)

    use_vulkan_training = selected_backend.name == "vulkan" and args.device in {"gpu", "auto"}
    use_backend_training = selected_backend.name in {"opencl", "cuda"} and args.device in {"gpu", "auto"}
    if args.device == "auto":
        if use_vulkan_training:
            print("Auto device selected: using Vulkan backend.")
        elif use_backend_training:
            print(f"Auto device selected: Vulkan unavailable, using {selected_backend.name} backend.")
        else:
            print("Auto device selected: no GPU backend available, falling back to CPU.")

    if use_vulkan_training:
        try:
            train_mlp_regression_gpu(
                x,
                y,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
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

    if use_backend_training:
        dataset = TensorDataset(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = BackendMLP(
            in_features=x.shape[1],
            hidden=16,
            out_features=y.shape[1],
            backend=selected_backend.name,
            seed=args.seed,
            optimizer="sgd",
            lr=args.lr,
            strict=False,
        )

        for epoch in range(args.epochs):
            epoch_loss = 0.0
            num_batches = 0
            should_log = (args.log_every > 0 and (epoch % args.log_every == 0)) or (epoch == args.epochs - 1)
            for xb_np, yb_np in loader:
                loss = model.train_step(xb_np, yb_np, lr=args.lr, return_loss=should_log)
                if should_log and loss is not None:
                    epoch_loss += float(loss)
                    num_batches += 1

            if should_log:
                avg_loss = epoch_loss / max(1, num_batches)
                print(f"Epoch {epoch}: loss={avg_loss:.6f}")

        if args.save:
            model.save(args.save)
            print(f"Saved model to: {args.save}")

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
        save_checkpoint(args.save, payload)
        print(f"Saved model to: {args.save}")

    print("Training done.")


if __name__ == "__main__":
    main()
