from abc import ABC, abstractmethod

import gc
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch_geometric.data import Batch, Data

from src.enums import Split

logger = logging.getLogger(__name__)


class ChunkedSplitDataset(IterableDataset, ABC):
    """Streams samples from flat chunk files, filtered by a split map.

    Each chunk file is loaded, its samples filtered to ``target_split`` via
    ``split_map``, yielded, then immediately deleted from memory.  RAM usage
    stays bounded to roughly one chunk at a time regardless of dataset size.

    Multi-worker support: chunk files are distributed across workers in a
    strided fashion (worker 0 gets files 0, N, 2N, ...; worker 1 gets
    1, N+1, ...) so no sample is yielded twice.

    Parameters
    ----------
    chunks_dir:
        Directory containing flat ``chunk_*.pt`` files (no train/val/test
        subdirs — all splits share the same files).
    split_map:
        Mapping of ``activity_id → split_name`` for the entire dataset.
        Loaded once in the main process and inherited by workers via fork.
    target_split:
        Which split to yield: ``'train'``, ``'val'``, or ``'test'``.
    shuffle:
        If True, shuffles chunk order and within-chunk sample order on
        every iteration.
    """

    def __init__(
        self,
        chunks_dir: Path,
        split_map: dict[int, str],
        target_split: Split,
        shuffle: bool = False,
        seed=20,
    ) -> None:
        self._rng = random.Random(seed)
        self._dir = Path(chunks_dir)
        self._split_map = split_map
        self._target_split = target_split
        self._shuffle = shuffle
        self._chunk_files = sorted(self._dir.glob("chunk_*.pt"))
        if not self._chunk_files:
            raise FileNotFoundError(f"No chunk files found in {self._dir}")
        self._total = sum(1 for v in split_map.values() if v == target_split)

    def __len__(self) -> int:
        return self._total

    @abstractmethod
    def _iter_chunk(self, chunk):
        pass

    def __iter__(self):
        worker_info = get_worker_info()
        chunk_files = list(self._chunk_files)

        if self._shuffle:
            self._rng.shuffle(chunk_files)

        if worker_info is not None:
            chunk_files = chunk_files[worker_info.id :: worker_info.num_workers]

        for chunk_file in chunk_files:
            chunk = torch.load(chunk_file, weights_only=False)
            samples = list(self._iter_chunk(chunk))
            if self._shuffle:
                self._rng.shuffle(samples)
            yield from samples
            del chunk, samples
            gc.collect()


class GraphSplitDataset(ChunkedSplitDataset):
    """Streams PyG ``Data`` objects from flat graph chunk files."""

    def _iter_chunk(self, chunk):
        # chunk is list[Data]; each Data has an activity_id attribute
        for data in chunk:
            if self._split_map.get(int(data.activity_id)) == self._target_split:
                yield data


class FingerprintSplitDataset(ChunkedSplitDataset):
    """Streams ``(fingerprint, label)`` tensor pairs from flat fingerprint chunk files."""

    def _iter_chunk(self, chunk):
        # chunk is (Tensor[N, n_bits], Tensor[N], Tensor[N]) = (X, y, activity_ids)
        x, y, activity_ids = chunk
        for xi, yi, aid in zip(x, y, activity_ids):
            if self._split_map.get(int(aid)) == self._target_split:
                yield xi, yi


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

    train_loader = DataLoader(train_ds, batch_size=batch_size, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, **kwargs)

    logger.info(
        "DataLoaders ready — train=%d, val=%d, test=%d samples",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    return train_loader, val_loader, test_loader


def create_graph_dataloaders(
    chunks_dir: Path,
    split_map: dict[int, str],
    batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from a flat graph chunk directory."""
    chunks_dir = Path(chunks_dir)
    train_ds = GraphSplitDataset(chunks_dir, split_map, Split.TRAIN, shuffle=True)
    val_ds = GraphSplitDataset(chunks_dir, split_map, Split.VAL)
    test_ds = GraphSplitDataset(chunks_dir, split_map, Split.TEST)
    return _make_loaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size,
        eval_batch_size,
        num_workers,
        Batch.from_data_list,
    )


def create_fp_dataloaders(
    chunks_dir: Path,
    split_map: dict[int, str],
    batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from a flat fingerprint chunk directory."""
    chunks_dir = Path(chunks_dir)
    train_ds = FingerprintSplitDataset(chunks_dir, split_map, Split.TRAIN, shuffle=True)
    val_ds = FingerprintSplitDataset(chunks_dir, split_map, Split.VAL)
    test_ds = FingerprintSplitDataset(chunks_dir, split_map, Split.TEST)
    return _make_loaders(
        train_ds, val_ds, test_ds, batch_size, eval_batch_size, num_workers
    )
