from __future__ import annotations

import gc
import json
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------------


class ChunkedSplitDataset(IterableDataset):
    """Streams samples from pre-split chunk files one file at a time.

    Each chunk file is loaded, its samples yielded via ``_iter_chunk``, then
    immediately deleted from memory.  RAM usage stays bounded to roughly one
    chunk at a time regardless of dataset size.

    Multi-worker support: chunk files are distributed across workers in a
    strided fashion (worker 0 gets files 0, N, 2N, ...; worker 1 gets
    1, N+1, ...) so no sample is yielded twice.

    Parameters
    ----------
    split_dir:
        Directory containing ``chunk_*.pt`` files and ``metadata.json``
        for one split (train, val, or test).
    shuffle:
        If True, shuffles chunk order and within-chunk sample order on
        every iteration.
    """

    def __init__(self, split_dir: Path, shuffle: bool = False) -> None:
        self._dir = Path(split_dir)
        self._shuffle = shuffle
        self._chunk_files = sorted(self._dir.glob("chunk_*.pt"))
        if not self._chunk_files:
            raise FileNotFoundError(f"No chunk files found in {self._dir}")

        meta_path = self._dir / "metadata.json"
        self._total: int | None = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            # support both key names used by the two builders
            self._total = meta.get("total_graphs") or meta.get("total_samples")

    def __len__(self) -> int:
        if self._total is None:
            raise TypeError(f"Dataset size unknown — metadata.json missing in {self._dir}")
        return self._total

    def _iter_chunk(self, chunk):
        """Yield individual samples from a loaded chunk. Override in subclasses."""
        raise NotImplementedError

    def __iter__(self):
        worker_info = get_worker_info()
        chunk_files = list(self._chunk_files)

        if self._shuffle:
            random.shuffle(chunk_files)

        if worker_info is not None:
            chunk_files = chunk_files[worker_info.id :: worker_info.num_workers]

        for chunk_file in chunk_files:
            chunk = torch.load(chunk_file, weights_only=False)
            samples = list(self._iter_chunk(chunk))
            if self._shuffle:
                random.shuffle(samples)
            yield from samples
            del chunk, samples
            gc.collect()


# ---------------------------------------------------------------------------
# Concrete dataset classes
# ---------------------------------------------------------------------------


class GraphSplitDataset(ChunkedSplitDataset):
    """Streams PyG ``Data`` objects from graph chunk files."""

    def _iter_chunk(self, chunk):
        yield from chunk  # chunk is list[Data]


class FingerprintSplitDataset(ChunkedSplitDataset):
    """Streams ``(fingerprint, label)`` tensor pairs from fingerprint chunk files."""

    def _iter_chunk(self, chunk):
        x, y = chunk  # chunk is (Tensor[N, n_bits], Tensor[N])
        yield from zip(x, y)


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------


def _make_loaders(
    train_ds: ChunkedSplitDataset,
    val_ds: ChunkedSplitDataset,
    test_ds: ChunkedSplitDataset,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    collate_fn=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    kwargs = dict(num_workers=num_workers)
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn

    train_loader = DataLoader(train_ds, batch_size=batch_size,      **kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=eval_batch_size, **kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=eval_batch_size, **kwargs)

    logger.info(
        "DataLoaders ready — train=%d, val=%d, test=%d samples",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_loader, val_loader, test_loader


def create_graph_dataloaders(
    split_graphs_dir: Path,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from a pre-split graph directory."""
    split_graphs_dir = Path(split_graphs_dir)
    train_ds = GraphSplitDataset(split_graphs_dir / "train", shuffle=True)
    val_ds   = GraphSplitDataset(split_graphs_dir / "val",   shuffle=False)
    test_ds  = GraphSplitDataset(split_graphs_dir / "test",  shuffle=False)
    return _make_loaders(train_ds, val_ds, test_ds, batch_size, eval_batch_size, num_workers, Batch.from_data_list)


def create_fp_dataloaders(
    split_fp_dir: Path,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from a pre-split fingerprint directory."""
    split_fp_dir = Path(split_fp_dir)
    train_ds = FingerprintSplitDataset(split_fp_dir / "train", shuffle=True)
    val_ds   = FingerprintSplitDataset(split_fp_dir / "val",   shuffle=False)
    test_ds  = FingerprintSplitDataset(split_fp_dir / "test",  shuffle=False)
    return _make_loaders(train_ds, val_ds, test_ds, batch_size, eval_batch_size, num_workers)
