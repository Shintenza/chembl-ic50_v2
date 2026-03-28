#!/usr/bin/env python3
"""
02_clean.py — Standardise SMILES and apply molecular filters to raw batches.

Usage
-----
    python scripts/02_clean.py

Reads raw Parquet files from ``config.PATHS['RAW_DIR']``, applies SMILES
standardisation and physicochemical filters, and writes cleaned Parquet
files to ``config.PATHS['CLEANED_DIR']``.  A per-batch statistics table is
printed at the end.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.cleaning import clean_all_batches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    raw_dir: Path = config.PATHS["RAW_DIR"]
    cleaned_dir: Path = config.PATHS["CLEANED_DIR"]

    logger.info("=== ChEMBL IC50 Cleaning ===")
    logger.info("Input  directory : %s", raw_dir)
    logger.info("Output directory : %s", cleaned_dir)
    logger.info("Filters:")
    logger.info(
        "  Atom count : [%d, %d]",
        config.CLEANING["MIN_ATOMS"],
        config.CLEANING["MAX_ATOMS"],
    )
    logger.info(
        "  Mol weight : [%.1f, %.1f] Da",
        config.CLEANING["MIN_MW"],
        config.CLEANING["MAX_MW"],
    )
    logger.info(
        "  pChEMBL    : [%.1f, %.1f]",
        config.CLEANING["MIN_PCHEMBL"],
        config.CLEANING["MAX_PCHEMBL"],
    )

    summary_df = clean_all_batches(raw_dir, cleaned_dir)

    if summary_df.empty:
        print("No batch files were processed. Is RAW_DIR populated?")
        return

    # Pretty-print the statistics table
    print()
    print("=" * 90)
    print("Cleaning Summary")
    print("=" * 90)
    print(
        f"{'Batch File':<25} {'Input':>10} {'Output':>10} "
        f"{'Bad SMILES':>12} {'Bad Mol':>10} {'pChEMBL OOR':>12}"
    )
    print("-" * 90)
    for _, row in summary_df.iterrows():
        print(
            f"{row['batch_file']:<25} {row['input_rows']:>10,} {row['output_rows']:>10,} "
            f"{row['dropped_invalid_smiles']:>12,} {row['dropped_invalid_mol']:>10,} "
            f"{row['dropped_pchembl_range']:>12,}"
        )
    print("-" * 90)
    totals = summary_df[
        [
            "input_rows",
            "output_rows",
            "dropped_invalid_smiles",
            "dropped_invalid_mol",
            "dropped_pchembl_range",
        ]
    ].sum()
    retention_pct = 100.0 * totals["output_rows"] / max(totals["input_rows"], 1)
    print(
        f"{'TOTAL':<25} {int(totals['input_rows']):>10,} {int(totals['output_rows']):>10,} "
        f"{int(totals['dropped_invalid_smiles']):>12,} {int(totals['dropped_invalid_mol']):>10,} "
        f"{int(totals['dropped_pchembl_range']):>12,}"
    )
    print("=" * 90)
    print(f"Retention rate: {retention_pct:.1f}%")
    print(f"Output directory: {cleaned_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
