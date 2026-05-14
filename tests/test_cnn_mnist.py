from __future__ import annotations
 
import os
import struct
import gzip
import urllib.request
 
import numpy as np
 
from rasptorch import functional as F
from rasptorch import vulkan_backend as vk
from rasptorch.data import DataLoader, TensorDataset
from rasptorch.nn import BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential
from rasptorch.optim import SGD
from rasptorch.tensor import Tensor
from rasptorch.train import Accuracy, classification_target_one_hot, fit
 
DEVICE = "cpu"
DATA_PATH = "./data"
MODEL_PATH = "mnist_model"
STATS_PATH = "mnist_stats.npz"
 
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SEED = 42
 
_MNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
 
 
def _download_mnist(root: str = DATA_PATH) -> None:
    raw = os.path.join(root, "MNIST", "raw")
    os.makedirs(raw, exist_ok=True)
    for fname in _MNIST_FILES:
        gz_path = os.path.join(raw, fname)
        out_path = os.path.join(raw, fname[:-3])
        if os.path.exists(out_path):
            continue
        print(f"Downloading {fname} …")
        urllib.request.urlretrieve(_MNIST_MIRROR + fname, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(gz_path)
 
 
def load_mnist(path: str = DATA_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _download_mnist(path)
 
    def _read(fname: str) -> np.ndarray:
        with open(fname, "rb") as f:
            magic = struct.unpack(">I", f.read(4))[0]
            if magic == 0x803:
                n, h, w = struct.unpack(">III", f.read(12))
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, h, w)
            if magic == 0x801:
                (n,) = struct.unpack(">I", f.read(4))
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(n)
            raise ValueError(f"Unknown magic number {magic:#010x} in {fname!r}")
 
    raw = os.path.join(path, "MNIST", "raw")
    x_train = _read(os.path.join(raw, "train-images-idx3-ubyte")).astype(np.float32) / 255.0
    y_train = _read(os.path.join(raw, "train-labels-idx1-ubyte")).astype(np.int64)
    x_test  = _read(os.path.join(raw, "t10k-images-idx3-ubyte")).astype(np.float32)  / 255.0
    y_test  = _read(os.path.join(raw, "t10k-labels-idx1-ubyte")).astype(np.int64)
 
    return x_train, y_train, x_test, y_test
 
 
def normalise(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)
 
 
def save_stats(mean: float, std: float) -> None:
    np.savez(STATS_PATH, mean=np.array(mean), std=np.array(std))
 
 
def load_stats() -> tuple[float, float]:
    data = np.load(STATS_PATH)
    return float(data["mean"]), float(data["std"])
 
 
def build_model() -> Sequential:
    return Sequential(
        Conv2d(1,  16, kernel_size=3, padding=1),
        BatchNorm2d(16),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2),
 
        Conv2d(16, 32, kernel_size=3, padding=1),
        BatchNorm2d(32),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2),
 
        Flatten(),
        Linear(32 * 7 * 7, 128),
        ReLU(),
        Linear(128, 10),
    )
 
 
def save_model(model: Sequential) -> None:
    params = {}
    for i, p in enumerate(model.parameters()):
        arr = p.data if p.device == "cpu" else vk.to_cpu(p._as_vkbuf())
        params[str(i)] = arr
    np.savez(MODEL_PATH, **params)
 
 
def load_model(model: Sequential) -> None:
    data = np.load(MODEL_PATH + ".npz")
    for i, p in enumerate(model.parameters()):
        arr = data[str(i)]
        if p.device == "gpu":
            vk.write(p._as_vkbuf(), arr)
        else:
            p.data[:] = arr
 
 
def predict(model: Sequential, x: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    x = normalise(x, mean, std)
    if x.ndim == 3:
        x = x[np.newaxis]
    inp = Tensor(x, requires_grad=False)
    logits = model(inp)
    arr = logits.data if logits.device == "cpu" else vk.to_cpu(logits._as_vkbuf())
    return np.argmax(arr, axis=1)
 
 
def train() -> None:
    np.random.seed(SEED)
    vk.init(strict=True)
 
    x_train, y_train, x_test, y_test = load_mnist()
 
    mean, std = float(x_train.mean()), float(x_train.std())
    save_stats(mean, std)
 
    train_loader = DataLoader(
        TensorDataset(normalise(x_train, mean, std), y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(normalise(x_test, mean, std), y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
 
    model = build_model().to(DEVICE)
    opt = SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
 
    fit(
        model,
        opt,
        train_loader,
        loss_fn=F.cross_entropy,
        device=DEVICE,
        epochs=EPOCHS,
        val_loader=val_loader,
        target_transform=classification_target_one_hot(10, device=DEVICE),
        metrics=[Accuracy()],
    )
 
    save_model(model)
 
 
def inference() -> None:
    vk.init(strict=True)
 
    mean, std = load_stats()
    _, _, x_test, y_test = load_mnist()
 
    model = build_model().to(DEVICE)
    load_model(model)
 
    all_preds: list[np.ndarray] = []
    for start in range(0, len(x_test), BATCH_SIZE * 2):
        batch = x_test[start : start + BATCH_SIZE * 2]
        all_preds.append(predict(model, batch, mean, std))
 
    preds = np.concatenate(all_preds)
    correct = int(np.sum(preds == y_test))
    print(f"Test accuracy: {correct / len(y_test) * 100:.2f}%  ({correct}/{len(y_test)} correct)")
 
 
if __name__ == "__main__":
    train()
    inference()
 