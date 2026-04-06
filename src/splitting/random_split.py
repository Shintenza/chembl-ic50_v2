from .utils import validate_splits
import logging
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.enums import Split

logger = logging.getLogger(__name__)


@validate_splits
def build_random_split_map(
    cleaned_dir: Path,
    frac_train: float,
    frac_val: float,
    seed: int,
) -> pd.DataFrame:
    """
    Build a random split map from cleaned parquet files.

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

    df = pd.concat(
        (
            pd.read_parquet(pq_path, engine="pyarrow", columns=["activity_id"])
            for pq_path in parquet_files
        ),
        ignore_index=True,
    )
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df)
    train_end = int(frac_train * n)
    val_end = train_end + int(frac_val * n)

    splits = (
        [Split.TRAIN] * train_end
        + [Split.VAL] * (val_end - train_end)
        + [Split.TEST] * (n - val_end)
    )

    return pd.DataFrame({"activity_id": df["activity_id"], "split": splits})
