"""Simple random train/val/test dataset splitting."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.enums import Split

logger = logging.getLogger(__name__)


def random_split(
    n: int,
    frac_train: float,
    frac_val: float,
    frac_test: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Randomly shuffle *n* indices and split into train / val / test.

    Parameters
    ----------
    n:
        Total number of samples.
    frac_train, frac_val, frac_test:
        Fractional sizes for each split (must sum to 1).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        ``(train_idx, val_idx, test_idx)`` — lists of integer indices.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, (
        "Fractions must sum to 1."
    )
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    logger.info(
        "random_split: train=%d, val=%d, test=%d (total=%d)",
        len(train_idx), len(val_idx), len(test_idx), n,
    )
    return train_idx, val_idx, test_idx


def build_random_split_map(
    cleaned_dir: Path,
    frac_train: float,
    frac_val: float,
    frac_test: float,
    seed: int,
) -> pd.DataFrame:
    """Build a random split map from cleaned parquet files.

    Reads only ``activity_id`` from each ``batch_*.parquet`` in *cleaned_dir*,
    randomly assigns each molecule to train / val / test.

    Parameters
    ----------
    cleaned_dir:
        Directory containing cleaned ``batch_*.parquet`` files.
    frac_train, frac_val, frac_test:
        Fractional sizes for each split (must sum to 1).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``activity_id``, ``split``.
    """
    cleaned_dir = Path(cleaned_dir)
    parquet_files = sorted(cleaned_dir.glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No cleaned parquet files found in {cleaned_dir}")

    chunks: list[pd.DataFrame] = []
    for pq_path in tqdm(parquet_files, desc="Loading cleaned data", unit="file"):
        df = pd.read_parquet(pq_path, engine="pyarrow", columns=["activity_id"])
        chunks.append(df)

    all_df = pd.concat(chunks, ignore_index=True)

    train_idx, val_idx, test_idx = random_split(
        n=len(all_df),
        frac_train=frac_train,
        frac_val=frac_val,
        frac_test=frac_test,
        seed=seed,
    )

    split_labels = pd.array([""] * len(all_df), dtype="object")
    split_labels[train_idx] = Split.TRAIN
    split_labels[val_idx]   = Split.VAL
    split_labels[test_idx]  = Split.TEST

    return pd.DataFrame({
        "activity_id": all_df["activity_id"],
        "split": split_labels,
    })
