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

    print(summary_df)


if __name__ == "__main__":
    main()
