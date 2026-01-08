from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np


class Dataset:
    """Minimal dataset base class, like torch.utils.data.Dataset."""

    def __len__(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def __getitem__(self, index):  # pragma: no cover - interface
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping one or more tensors/arrays of equal length."""

    def __init__(self, *arrays) -> None:
        if not arrays:
            raise ValueError("TensorDataset requires at least one tensor/array")
        length = len(arrays[0])
        for arr in arrays[1:]:
            if len(arr) != length:
                raise ValueError("All tensors/arrays must have the same length")
        self._arrays = arrays
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        return tuple(arr[index] for arr in self._arrays)


class DataLoader(Iterable):
    """Simple DataLoader with batching and optional shuffling."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        batch: List[int] = []
        for idx in indices:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield self._fetch_batch(batch)
                batch = []
        if batch:
            yield self._fetch_batch(batch)

    def _fetch_batch(self, indices: Sequence[int]):
        batch_items = [self.dataset[i] for i in indices]
        # Assume each item is a tuple of tensors/arrays; transpose batch dim
        first = batch_items[0]
        if not isinstance(first, tuple):
            # Single tensor/array case
            stacked = np.stack(batch_items, axis=0)
            return stacked
        transposed = list(zip(*batch_items))
        return tuple(np.stack(part, axis=0) for part in transposed)
