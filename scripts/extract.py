#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.extraction import extract_batches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    raw_dir: Path = config.PATHS["RAW_DIR"]
    batch_size: int = config.EXTRACTION["BATCH_SIZE"]

    logger.info("=== ChEMBL IC50 Extraction ===")
    logger.info("Output directory : %s", raw_dir)
    logger.info("Batch size        : %d", batch_size)
    logger.info("DB host           : %s", config.DB["host"])
    logger.info("DB name           : %s", config.DB["dbname"])

    total_rows = extract_batches(
        db_config=config.DB, output_dir=raw_dir, batch_size=batch_size, start_offset=0
    )

    logger.info(f"Extracted {total_rows} rows")


if __name__ == "__main__":
    main()
