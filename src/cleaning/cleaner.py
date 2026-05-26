from src.cleaning.clean_utils import get_smiles_median_mapping, clean_batch
import logging
import sys
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from .smiles_utils import standardize_smiles, is_valid_molecule

logger = logging.getLogger(__name__)


def clean_all_batches(raw_dir: Path, cleaned_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    smiles_mapping = get_smiles_median_mapping(
        raw_dir, "canonical_smiles", "pchembl_value", 1.0
    )

    batch_files = sorted(raw_dir.glob("batch_*.parquet"))
    if not batch_files:
        logger.warning("No batch files found in %s", raw_dir)
        return pd.DataFrame()

    records: list[dict] = []

    for batch_path in tqdm(batch_files, desc="Cleaning batches", unit="file"):
        out_path = cleaned_dir / batch_path.name
        stats = clean_batch(batch_path, out_path, smiles_mapping)
        stats["batch_file"] = batch_path.name
        records.append(stats)

    summary_df = pd.DataFrame(records)[
        [
            "batch_file",
            "input_rows",
            "output_rows",
            "dropped_invalid_smiles",
            "dropped_invalid_mol",
            "dropped_pchembl_range",
        ]
    ]
    return summary_df
