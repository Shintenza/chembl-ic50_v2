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
# IterableDataset
# ---------------------------------------------------------------------------


class SplitGraphDataset(IterableDataset):
    """Streams graphs from pre-split chunk files one file at a time.

    Each chunk file is loaded, its graphs yielded to the DataLoader, then
    immediately deleted from memory.  This keeps RAM usage bounded to
    roughly one chunk at a time regardless of dataset size.

    Multi-worker support: chunk files are distributed across workers in a
    strided fashion (worker 0 gets files 0, N, 2N, ...; worker 1 gets
    1, N+1, ...) so no graph is yielded twice and all workers stay busy.

    Parameters
    ----------
    split_dir:
        Directory containing ``chunk_*.pt`` files and ``metadata.json``
        for one split (train, val, or test).
    shuffle:
        If True, shuffles the chunk order and within-chunk graph order on
        every iteration (each epoch gets a different ordering).
    """

    def __init__(self, split_dir: Path, shuffle: bool = False) -> None:
        self._dir = Path(split_dir)
        self._shuffle = shuffle
        self._chunk_files = sorted(self._dir.glob("chunk_*.pt"))
        if not self._chunk_files:
            raise FileNotFoundError(f"No chunk files found in {self._dir}")

        meta_path = self._dir / "metadata.json"
        self._total_graphs: int | None = None
        if meta_path.exists():
            self._total_graphs = json.loads(meta_path.read_text())["total_graphs"]

    def __len__(self) -> int:
        if self._total_graphs is None:
            raise TypeError(f"Dataset size unknown — metadata.json missing in {self._dir}")
        return self._total_graphs

    def __iter__(self):
        worker_info = get_worker_info()
        chunk_files = list(self._chunk_files)

        if self._shuffle:
            random.shuffle(chunk_files)

        # Distribute chunk files evenly across workers.
        # Each worker takes every Nth file (strided), so:
        #   worker 0 → files [0, N, 2N, ...]
        #   worker 1 → files [1, N+1, 2N+1, ...]
        if worker_info is not None:
            chunk_files = chunk_files[worker_info.id :: worker_info.num_workers]

        for chunk_file in chunk_files:
            graphs: list[Data] = torch.load(chunk_file, weights_only=False)
            if self._shuffle:
                random.shuffle(graphs)
            yield from graphs
            del graphs  # release memory before loading next chunk
            gc.collect()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_dataloaders(
    split_graphs_dir: Path,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders from a pre-split graph directory.

    Parameters
    ----------
    split_graphs_dir:
        Directory produced by ``reorganize_by_split`` — contains ``train/``,
        ``val/``, and ``test/`` subdirectories, each with ``chunk_*.pt`` files.
    batch_size:
        Mini-batch size for the training loader.
    eval_batch_size:
        Mini-batch size for validation and test loaders.
    num_workers:
        Number of worker processes for the inner DataLoader.
    """
    split_graphs_dir = Path(split_graphs_dir)

    train_ds = SplitGraphDataset(split_graphs_dir / "train", shuffle=True)
    val_ds   = SplitGraphDataset(split_graphs_dir / "val",   shuffle=False)
    test_ds  = SplitGraphDataset(split_graphs_dir / "test",  shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,      collate_fn=Batch.from_data_list, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=eval_batch_size, collate_fn=Batch.from_data_list, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=eval_batch_size, collate_fn=Batch.from_data_list, num_workers=num_workers)

    logger.info("DataLoaders ready — train=%d, val=%d, test=%d graphs",
                len(train_ds), len(val_ds), len(test_ds))

    return train_loader, val_loader, test_loader
