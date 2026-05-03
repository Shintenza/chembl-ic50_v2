import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, cast

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)

BuilderFn = Callable[[str, float, int], object | None]


def build_features(
    cleaned_dir: Path, output_dir: Path, chunk_size: int, builder_fn: BuilderFn
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(cleaned_dir.glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No cleaned parquet files found in {cleaned_dir}")

    buffer = []
    chunk_count = 0
    total = 0

    for parquet_path in tqdm(parquet_files, desc="Building features", unit="file"):
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False, unit="mol"):
            smiles = str(row["canonical_smiles"])
            pic50 = cast(float, row["pchembl_value"])
            activity_id = cast(int, row["activity_id"])

            data_piece = builder_fn(smiles, pic50, activity_id)
            if data_piece is None:
                continue

            buffer.append(data_piece)
            total += 1

            if len(buffer) >= chunk_size:
                save_batch(buffer, output_dir, chunk_count)
                chunk_count += 1
                buffer = []

    if len(buffer) > 0:
        save_batch(buffer, output_dir, chunk_count)

    meta = {"total": total}
    (output_dir / "metadata.json").write_text(json.dumps(meta))
    logger.info("Total features written: %d", total)

    return total


def save_batch(items: list[object], out_dir: Path, n: int) -> None:
    torch.save(items, out_dir / f"chunk_{n:04d}.pt")
