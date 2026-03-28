#!/usr/bin/env python3
"""
01_extract.py — Extract IC50 records from ChEMBL in paginated batches.

Usage
-----
    python scripts/01_extract.py
    python scripts/01_extract.py --start-offset 900000

Each batch is written as a Parquet file to the directory specified by
``config.PATHS['RAW_DIR']``.  Re-run with ``--start-offset`` to resume an
interrupted extraction without re-downloading already-saved batches.
"""

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.extraction import extract_batches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract IC50 data from ChEMBL PostgreSQL database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Row offset at which to begin extraction. "
            "Use to resume after an interrupted run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir: Path = config.PATHS["RAW_DIR"]
    batch_size: int = config.EXTRACTION["BATCH_SIZE"]

    logger.info("=== ChEMBL IC50 Extraction ===")
    logger.info("Output directory : %s", raw_dir)
    logger.info("Batch size        : %d", batch_size)
    logger.info("Start offset      : %d", args.start_offset)
    logger.info("DB host           : %s", config.DB["host"])
    logger.info("DB name           : %s", config.DB["dbname"])

    total_rows = extract_batches(
        db_config=config.DB,
        output_dir=raw_dir,
        batch_size=batch_size,
        start_offset=args.start_offset,
    )

    print()
    print("=" * 50)
    print(f"Extraction complete.")
    print(f"  Total rows extracted this run : {total_rows:,}")
    print(f"  Output directory              : {raw_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
